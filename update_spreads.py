import json
import io
import os
import logging
from datetime import datetime, date, timedelta, timezone

import numpy as np
from scipy.stats import norm
import yfinance as yf

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# -------------------- Config & Setup --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Ensure these env vars are set in your GitHub Action
FILE_ID = os.environ.get("GDRIVE_FILE_ID")
CREDS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

# -------------------- Google Drive --------------------
def get_drive_service():
    if not CREDS_JSON or not FILE_ID:
        raise ValueError("Missing Environment Variables for Google Drive.")
    creds = Credentials.from_service_account_info(
        json.loads(CREDS_JSON),
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def download_json(service):
    logger.info("Downloading JSON from Drive...")
    fh = io.BytesIO()
    request = service.files().get_media(fileId=FILE_ID)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    try:
        return json.load(fh)
    except json.JSONDecodeError:
        return [] # Return empty list if file is empty

def upload_json(service, data):
    logger.info("Uploading updated JSON to Drive...")
    fh = io.BytesIO(json.dumps(data, indent=2).encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=False)
    service.files().update(fileId=FILE_ID, media_body=media).execute()

# -------------------- Math & Financials --------------------
def days_to_expiry(expiry_str):
    if not expiry_str: 
        return 0
    try:
        expiry = date.fromisoformat(expiry_str)
    except ValueError:
        return 0
    return max((expiry - date.today()).days, 0)

def bsm_delta(option_type, S, K, T, r, sigma):
    """
    S: Spot Price
    K: Strike Price
    T: Time to expiry (in years)
    r: Risk-free rate (decimal, e.g., 0.05)
    sigma: Implied Volatility (decimal, e.g., 0.20)
    """
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == "put":
        return norm.cdf(d1) - 1
    # call
    return norm.cdf(d1)

def get_option_data(ticker, expiration_str, short_strike, long_strike):
    """
    Fetches the specific option chain for the expiration date.
    Returns: short_price, short_iv, long_price
    """
    try:
        t = yf.Ticker(ticker)
        
        # yfinance requires expiration strings to match their available dates exactly.
        # We try to find a close match if the user input formatted it differently.
        avail_dates = t.options
        if expiration_str not in avail_dates:
            logger.warning(f"{ticker}: Expiration {expiration_str} not found in {avail_dates}")
            return None, None, None

        chain = t.option_chain(expiration_str)
        puts = chain.puts

        # Filter for strikes
        short_row = puts[puts['strike'] == short_strike]
        long_row = puts[puts['strike'] == long_strike]

        if short_row.empty or long_row.empty:
            logger.warning(f"{ticker}: Strikes {short_strike}/{long_strike} not found.")
            return None, None, None

        # Extract data (using lastPrice as a proxy for mark, or (bid+ask)/2 if you prefer)
        # Using lastPrice is often more stable for free data, though less accurate for illiquid options.
        short_price = float(short_row['lastPrice'].values[0])
        short_iv = float(short_row['impliedVolatility'].values[0]) # Decimal format from yf
        long_price = float(long_row['lastPrice'].values[0])

        return short_price, short_iv, long_price

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None

# -------------------- Core Update Logic --------------------
def update_trade(trade):
    # 1. Basic Setup
    ticker = trade.get("ticker")
    expiry_str = trade.get("expiration")
    
    try:
        short_strike = float(trade.get("short_strike", 0))
        long_strike = float(trade.get("long_strike", 0))
        credit_received = float(trade.get("credit", 0))
    except ValueError:
        return trade # Skip malformed trades

    width = abs(short_strike - long_strike)
    max_loss = width - credit_received

    # 2. Fetch Underlying Price
    try:
        ticker_obj = yf.Ticker(ticker)
        # fast_info is usually faster/more reliable for current price
        current_price = ticker_obj.fast_info.get("last_price") or ticker_obj.history(period="1d")['Close'].iloc[-1]
    except Exception:
        current_price = None

    # 3. Fetch Option Legs
    short_price, short_iv, long_price = get_option_data(ticker, expiry_str, short_strike, long_strike)

    # 4. Calculate Delta (Short Leg)
    # yfinance IV is usually decimal (0.25). BSM needs decimal.
    # Time must be in years.
    delta = None
    dte = days_to_expiry(expiry_str)
    
    if current_price and short_iv and dte > 0:
        T_years = dte / 365.0
        # Assuming 5% risk free rate
        delta = bsm_delta("put", current_price, short_strike, T_years, 0.05, short_iv)

    # 5. Calculate Metrics
    # Profit % = (Credit - CurrentSpreadValue) / MaxGain
    # CurrentSpreadValue = ShortOption - LongOption
    profit_pct = None
    spread_val_pct = None

    if short_price is not None and long_price is not None:
        current_spread_cost = short_price - long_price
        
        # Calculate Profit % of Max Gain
        if credit_received > 0:
            current_profit = credit_received - current_spread_cost
            profit_pct = (current_profit / credit_received) * 100
            
        # Calculate Spread Value % (Cost to close / Credit Received) ? 
        # OR (Cost to close / Max Loss)? 
        # User requirement: "Must remain below 150-200% of the credit received"
        if credit_received > 0:
            spread_val_pct = (current_spread_cost / credit_received) * 100

    # 6. Check Rules
    rule_violations = { "other_rules": False, "iv_rule": False }
    
    # Delta Rule (< 0.40) (Check abs value because put delta is negative)
    if delta is not None and abs(delta) >= 0.40:
        rule_violations["other_rules"] = True
    
    # Spread Value Rule (< 150% of credit)
    if spread_val_pct is not None and spread_val_pct >= 150:
        rule_violations["other_rules"] = True
        
    # DTE Rule (> 7 days)
    if dte <= 7:
        rule_violations["other_rules"] = True

    # 7. Update Trade Object
    # We update the 'cached' dictionary. App reads from here.
    trade["cached"] = {
        "current_price": current_price,
        "abs_delta": abs(delta) if delta is not None else None, 
        "short_option_price": short_price,
        "long_option_price": long_price,
        "current_profit_percent": profit_pct,
        "spread_value_percent": spread_val_pct,
        "rule_violations": rule_violations,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    # 8. Update History (PnL Tracking)
    # Logic: Only add one entry per day.
    today_str = date.today().isoformat()
    if "pnl_history" not in trade:
        trade["pnl_history"] = []

    # Check if today exists
    existing_entry = next((item for item in trade["pnl_history"] if item["date"] == today_str), None)
    
    if profit_pct is not None:
        if existing_entry:
            # Update today's entry with latest run
            existing_entry["profit"] = profit_pct
            existing_entry["dte"] = dte
        else:
            # Create new entry
            trade["pnl_history"].append({
                "date": today_str,
                "dte": dte,
                "profit": profit_pct
            })

    return trade

def main():
    try:
        service = get_drive_service()
        trades = download_json(service)
        
        if not trades:
            logger.info("No trades found in JSON.")
            return

        updated_trades = []
        for trade in trades:
            logger.info(f"Updating {trade.get('ticker')}...")
            updated_trades.append(update_trade(trade))

        upload_json(service, updated_trades)
        logger.info("Update complete.")
        
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}")
        raise

if __name__ == "__main__":
    main()
