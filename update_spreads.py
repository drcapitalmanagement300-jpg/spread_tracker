import json
import io
import os
import logging
import requests
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

FILE_ID = os.environ.get("GDRIVE_FILE_ID")
CREDS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# -------------------- Discord Notification --------------------
def send_discord_alert(ticker, description, color=15158332):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord Webhook URL not set. Skipping notification.")
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
    """
    Sends a daily status summary at approx 9:00 AM Eastern.
    Uses UTC 13:00/14:00 window as a trigger.
    """
    now_utc = datetime.now(timezone.utc)
    # 9:00 AM ET is 13:00 UTC (Daylight) or 14:00 UTC (Standard)
    # We trigger if it's within the 13:00 or 14:00 hour
    if now_utc.hour in [13, 14]:
        # Check a 'global' flag in the first trade object to see if heartbeat sent today
        # If no trades exist, we can't store state, but main() handles that
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
            send_discord_alert("System Heartbeat", summary, color=3447003) # Blue
            
            # Save state to the first trade to prevent multiple heartbeats in the same hour
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

def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "put":
        return norm.cdf(d1) - 1
    return norm.cdf(d1)

def get_option_data(ticker, expiration_str, short_strike, long_strike):
    try:
        t = yf.Ticker(ticker)
        avail_dates = t.options
        if expiration_str not in avail_dates:
            return None, None, None

        chain = t.option_chain(expiration_str)
        puts = chain.puts
        short_row = puts[puts['strike'] == short_strike]
        long_row = puts[puts['strike'] == long_strike]

        if short_row.empty or long_row.empty:
            return None, None, None

        short_price = float(short_row['lastPrice'].values[0])
        short_iv = float(short_row['impliedVolatility'].values[0]) 
        long_price = float(long_row['lastPrice'].values[0])

        return short_price, short_iv, long_price
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None

# -------------------- Core Update Logic --------------------
def update_trade(trade):
    ticker = trade.get("ticker")
    expiry_str = trade.get("expiration")
    
    try:
        short_strike = float(trade.get("short_strike", 0))
        long_strike = float(trade.get("long_strike", 0))
        credit_received = float(trade.get("credit", 0))
    except ValueError:
        return trade 

    try:
        ticker_obj = yf.Ticker(ticker)
        current_price = ticker_obj.fast_info.get("last_price") or ticker_obj.history(period="1d")['Close'].iloc[-1]
    except Exception:
        current_price = None

    short_price, short_iv, long_price = get_option_data(ticker, expiry_str, short_strike, long_strike)
    dte = days_to_expiry(expiry_str)

    delta = None
    if current_price and short_iv and dte > 0:
        delta = bsm_delta("put", current_price, short_strike, dte/365.0, 0.05, short_iv)

    profit_pct = None
    spread_val_pct = None
    if short_price is not None and long_price is not None and credit_received > 0:
        current_spread_cost = short_price - long_price
        profit_pct = ((credit_received - current_spread_cost) / credit_received) * 100
        spread_val_pct = (current_spread_cost / credit_received) * 100

    rule_violations = { "other_rules": False, "iv_rule": False }
    notif_msg = ""
    notif_color = 15158332 # Red

    if profit_pct is not None and profit_pct >= 50:
        notif_msg = f"✅ **Target Reached**: {profit_pct:.1f}% profit."
        notif_color = 3066993 # Green
    elif delta is not None and abs(delta) >= 0.40:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Delta Breach**: {abs(delta):.2f} (Limit 0.40)"
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
        "abs_delta": abs(delta) if delta is not None else None, 
        "short_option_price": short_price,
        "long_option_price": long_price,
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
        else:
            trade["pnl_history"].append({"date": today_str, "dte": dte, "profit": profit_pct})

    return trade

def main():
    try:
        service = get_drive_service()
        trades = download_json(service)
        if not trades: return
        
        updated_trades = [update_trade(trade) for trade in trades]
        
        # Patch: Handle the Heartbeat Check
        handle_heartbeat(updated_trades)
        
        upload_json(service, updated_trades)
        logger.info("Update complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
