import json
import io
import os
import time
import logging
import requests
from datetime import datetime, date, timedelta, timezone

import numpy as np
from scipy.stats import norm
import yfinance as yf

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# Import Lock Manager (Safe fallback)
try:
    from google_drive import DriveLockManager
except ImportError:
    logging.warning("Could not import DriveLockManager. Running without locks.")
    DriveLockManager = None

# -------------------- Config & Setup --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

FILE_ID = os.environ.get("GDRIVE_FILE_ID")
CREDS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# -------------------- Data Reliability Layer --------------------
class MarketDataManager:
    """
    Handles fetching market data with caching and retry logic.
    Prevents redundant API calls when multiple trades share the same ticker/expiry.
    """
    def __init__(self):
        self._price_cache = {}    # {ticker: (price, change_pct)}
        self._chain_cache = {}    # {(ticker, expiry_date): option_chain_obj}
        self._ticker_objs = {}    # {ticker: yf.Ticker}

    def _get_yf_ticker(self, ticker):
        if ticker not in self._ticker_objs:
            self._ticker_objs[ticker] = yf.Ticker(ticker)
        return self._ticker_objs[ticker]

    def get_price_data(self, ticker):
        """
        Fetches current price AND daily percentage change.
        Returns: (price, change_percent)
        """
        # Check cache
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        ticker_obj = self._get_yf_ticker(ticker)
        price = None
        change_pct = 0.0
        
        # Retry loop
        for attempt in range(3):
            try:
                # fast_info provides both last_price and previous_close efficiently
                info = ticker_obj.fast_info
                price = info.get("last_price")
                prev_close = info.get("previous_close")
                
                # Check if data is valid
                if price is not None and prev_close is not None and prev_close > 0:
                    change_pct = ((price - prev_close) / prev_close) * 100
                    break
                
                # Fallback to history if fast_info fails (e.g. sometimes on indices)
                if price is None:
                    hist = ticker_obj.history(period="2d")
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                        if len(hist) >= 2:
                            prev_close = hist['Close'].iloc[-2]
                            change_pct = ((price - prev_close) / prev_close) * 100
                
                if price is not None:
                    break
            except Exception as e:
                logger.warning(f"[Attempt {attempt+1}/3] Price data failed for {ticker}: {e}")
                time.sleep(2 * (attempt + 1))

        if price:
            self._price_cache[ticker] = (price, change_pct)
            return price, change_pct
        
        return None, None

    def get_chain(self, ticker, expiration):
        """Fetches option chain with caching and retries."""
        cache_key = (ticker, expiration)
        if cache_key in self._chain_cache:
            return self._chain_cache[cache_key]

        ticker_obj = self._get_yf_ticker(ticker)
        chain = None

        # Quick check if date is valid
        try:
            if expiration not in ticker_obj.options:
                return None
        except Exception:
            pass 

        # Retry loop
        for attempt in range(3):
            try:
                chain = ticker_obj.option_chain(expiration)
                if chain is not None:
                    break
            except Exception as e:
                logger.warning(f"[Attempt {attempt+1}/3] Chain fetch failed for {ticker} {expiration}: {e}")
                time.sleep(2 * (attempt + 1))

        if chain:
            self._chain_cache[cache_key] = chain
            return chain
        
        return None

# -------------------- Discord Notification --------------------
def send_discord_alert(ticker, description, color=15158332):
    if not DISCORD_WEBHOOK_URL:
        return

    payload = {
        "username": "DR Capital Monitor",
        "embeds": [{
            "title": f"🔔 {ticker}",
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "DR Capital Portfolio System"}
        }]
    }

    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {e}")

# -------------------- Heartbeat Logic --------------------
def handle_heartbeat(updated_trades):
    now_utc = datetime.now(timezone.utc)
    # Trigger window: 13:00 or 14:00 UTC
    if now_utc.hour in [13, 14]:
        if not updated_trades: return
        
        last_hb = updated_trades[0].get("last_heartbeat_date")
        today_str = date.today().isoformat()
        
        if last_hb != today_str:
            total = len(updated_trades)
            breaches = sum(1 for t in updated_trades if t.get("cached", {}).get("rule_violations", {}).get("other_rules"))
            
            summary = (
                f"☀️ **Morning System Check (9:00 AM ET)**\n"
                f"✅ **Status:** Online\n"
                f"📊 **Positions:** {total}\n"
                f"🚨 **Breaches:** {breaches}\n"
                f"All systems operational."
            )
            send_discord_alert("System Heartbeat", summary, color=3447003) 
            updated_trades[0]["last_heartbeat_date"] = today_str

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
        return []

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

def calculate_greeks(option_type, S, K, T, r, sigma):
    """
    Calculates Delta, Gamma, and Theta (Daily) using Black-Scholes.
    """
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    if option_type == "call":
        delta = cdf_d1
    else: # put
        delta = cdf_d1 - 1

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))

    # Theta (Annual)
    term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta_annual = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else: # put
        theta_annual = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

    theta_daily = theta_annual / 365.0

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_daily
    }

def get_option_data(data_manager, ticker, expiration_str, short_strike, long_strike):
    """
    Uses the MarketDataManager to retrieve cached chain data.
    """
    try:
        chain = data_manager.get_chain(ticker, expiration_str)
        if chain is None:
            return None, None, None, None

        puts = chain.puts
        
        short_row = puts[puts['strike'] == short_strike]
        long_row = puts[puts['strike'] == long_strike]

        if short_row.empty or long_row.empty:
            return None, None, None, None

        short_price = float(short_row['lastPrice'].values[0])
        long_price = float(long_row['lastPrice'].values[0])
        short_iv = float(short_row['impliedVolatility'].values[0])
        long_iv = float(long_row['impliedVolatility'].values[0])

        return short_price, long_price, short_iv, long_iv
    except Exception as e:
        logger.error(f"Error processing option data for {ticker}: {e}")
        return None, None, None, None

# -------------------- Core Update Logic --------------------
def update_trade(trade, data_manager):
    ticker = trade.get("ticker")
    expiry_str = trade.get("expiration")
    
    try:
        short_strike = float(trade.get("short_strike", 0))
        long_strike = float(trade.get("long_strike", 0))
        credit_received = float(trade.get("credit", 0))
    except ValueError:
        return trade 

    # 1. Fetch Price & Change (Cached/Retried)
    current_price, day_change_pct = data_manager.get_price_data(ticker)

    # 2. Fetch Option Data (Cached/Retried)
    short_price, long_price, short_iv, long_iv = get_option_data(
        data_manager, ticker, expiry_str, short_strike, long_strike
    )
    
    dte = days_to_expiry(expiry_str)
    T_years = dte / 365.0

    # 3. Calculate Greeks
    short_leg_greeks = {"delta": 0, "gamma": 0, "theta": 0}
    long_leg_greeks = {"delta": 0, "gamma": 0, "theta": 0}
    
    if current_price and dte > 0:
        if short_iv:
            short_leg_greeks = calculate_greeks("put", current_price, short_strike, T_years, 0.05, short_iv)
        if long_iv:
            long_leg_greeks = calculate_greeks("put", current_price, long_strike, T_years, 0.05, long_iv)

    # Net Position Greeks (Short Put Spread = Short the Short, Long the Long)
    net_delta = (-1 * short_leg_greeks["delta"]) + (1 * long_leg_greeks["delta"])
    net_gamma = (-1 * short_leg_greeks["gamma"]) + (1 * long_leg_greeks["gamma"])
    net_theta = (-1 * short_leg_greeks["theta"]) + (1 * long_leg_greeks["theta"])

    # 4. PnL Stats
    profit_pct = None
    spread_val_pct = None
    if short_price is not None and long_price is not None and credit_received > 0:
        current_spread_cost = short_price - long_price
        profit_pct = ((credit_received - current_spread_cost) / credit_received) * 100
        spread_val_pct = (current_spread_cost / credit_received) * 100

    # 5. Rules & Alerts
    rule_violations = { "other_rules": False, "iv_rule": False }
    notif_msg = ""
    notif_color = 15158332 # Red

    short_abs_delta = abs(short_leg_greeks["delta"]) # For assignment risk

    if profit_pct is not None and profit_pct >= 50:
        notif_msg = f"✅ **Target Reached**: {profit_pct:.1f}% profit."
        notif_color = 3066993 # Green
    elif short_abs_delta >= 0.40:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Delta Breach**: {short_abs_delta:.2f} (Limit 0.40)"
    elif spread_val_pct is not None and spread_val_pct >= 150:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Spread Value High**: {spread_val_pct:.0f}% of credit."
    elif dte <= 7:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Low DTE**: Only {dte} days remaining."

    if notif_msg:
        last_sent_str = trade.get("last_alert_sent")
        should_send = True
        if last_sent_str:
            last_sent = datetime.fromisoformat(last_sent_str).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_sent < timedelta(hours=20):
                should_send = False
        
        if should_send:
            send_discord_alert(f"Position Alert: {ticker}", notif_msg, notif_color)
            trade["last_alert_sent"] = datetime.now(timezone.utc).isoformat()

    trade["cached"] = {
        "current_price": current_price,
        "day_change_percent": day_change_pct, # <--- Stored here
        "short_option_price": short_price,
        "long_option_price": long_price,
        "abs_delta": short_abs_delta, 
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_theta": net_theta,
        "current_profit_percent": profit_pct,
        "spread_value_percent": spread_val_pct,
        "rule_violations": rule_violations,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    today_str = date.today().isoformat()
    if "pnl_history" not in trade: trade["pnl_history"] = []
    existing_entry = next((item for item in trade["pnl_history"] if item["date"] == today_str), None)
    if profit_pct is not None:
        if existing_entry:
            existing_entry["profit"] = profit_pct
            existing_entry["dte"] = dte
            existing_entry["net_theta"] = net_theta
        else:
            trade["pnl_history"].append({
                "date": today_str, 
                "dte": dte, 
                "profit": profit_pct,
                "net_theta": net_theta
            })

    return trade

def main():
    try:
        service = get_drive_service()
        
        lock = None
        if DriveLockManager:
            lock = DriveLockManager(service, FILE_ID)
            lock.acquire()
        
        try:
            trades = download_json(service)
            if not trades: return
            
            market_data = MarketDataManager()
            updated_trades = [update_trade(t, market_data) for t in trades]
            
            handle_heartbeat(updated_trades)
            upload_json(service, updated_trades)
            logger.info("Update complete.")
            
        except Exception as inner_e:
            logger.error(f"Error during update processing: {inner_e}")
            raise
        finally:
            if lock:
                lock.release()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
