import json
import io
import os
from datetime import datetime, date, timezone

import numpy as np
from scipy.stats import norm
import yfinance as yf

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# -------------------- Google Drive --------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
FILE_ID = os.environ["GDRIVE_FILE_ID"]

def get_drive_service():
    creds = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]),
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def download_json(service):
    fh = io.BytesIO()
    request = service.files().get_media(fileId=FILE_ID)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

def upload_json(service, data):
    fh = io.BytesIO(json.dumps(data, indent=2).encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=False)
    service.files().update(fileId=FILE_ID, media_body=media).execute()

# -------------------- Math (lifted from app) --------------------
def days_to_expiry(expiry):
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    return max((expiry - date.today()).days, 0)

def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "put":
        return norm.cdf(d1) - 1
    return norm.cdf(d1)

def get_option_chain(ticker, expiration):
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiration)
    return chain.puts

def get_leg_data(ticker, expiration, strike):
    puts = get_option_chain(ticker, expiration)
    row = puts[puts["strike"] == strike]
    if row.empty:
        return None, None
    price = float(row["lastPrice"].values[0])
    iv = float(row["impliedVolatility"].values[0]) * 100
    return price, iv

def compute_spread_value(short_price, long_price, width, credit):
    if short_price is None or long_price is None:
        return None
    max_loss = width - credit
    if max_loss <= 0:
        return None
    spread_mark = short_price - long_price
    return (spread_mark / max_loss) * 100

def compute_profit(short_price, long_price, credit):
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    profit = credit - spread_value
    return max(0, min((profit / credit) * 100, 100))

# -------------------- Core Update --------------------
def update_trade(trade):
    ticker = trade["ticker"]
    expiry = trade["expiration"]
    if isinstance(expiry, date):
        expiry_str = expiry.isoformat()
    else:
        expiry_str = expiry

    short_strike = float(trade["short_strike"])
    long_strike = float(trade["long_strike"])
    credit = float(trade["credit"])
    width = abs(short_strike - long_strike)

    # Price
    price = float(yf.Ticker(ticker).fast_info["last_price"])

    # Option legs
    short_price, iv = get_leg_data(ticker, expiry_str, short_strike)
    long_price, _ = get_leg_data(ticker, expiry_str, long_strike)

    # Delta (short leg only — EXACT match to app)
    delta = None
    if price and iv:
        T = days_to_expiry(expiry) / 365
        delta = bsm_delta("put", price, short_strike, T, 0.05, iv / 100)

    # Metrics
    profit_pct = compute_profit(short_price, long_price, credit)
    spread_value_pct = compute_spread_value(short_price, long_price, width, credit)
    dte = days_to_expiry(expiry)

    # Rules
    rule_violations = {
        "other_rules": False,
        "iv_rule": False
    }

    if delta is not None and abs(delta) >= 0.40:
        rule_violations["other_rules"] = True
    if spread_value_pct is not None and spread_value_pct >= 150:
        rule_violations["other_rules"] = True
    if dte <= 7:
        rule_violations["other_rules"] = True
    if trade.get("entry_iv") and iv and iv > trade["entry_iv"]:
        rule_violations["iv_rule"] = True

    # Update cached
    trade["cached"] = {
        "current_price": price,
        "delta": delta,
        "short_option_price": short_price,
        "long_option_price": long_price,
        "current_profit_percent": profit_pct,
        "spread_value_percent": spread_value_pct,
        "rule_violations": rule_violations
    }

    # Update PnL history (once per day)
    today = date.today().isoformat()
    if "pnl_history" not in trade:
        trade["pnl_history"] = []

    if not any(p["date"] == today for p in trade["pnl_history"]):
        trade["pnl_history"].append({
            "date": today,
            "dte": dte,
            "profit": profit_pct
        })

    trade["last_update"] = datetime.now(timezone.utc).isoformat()
    return trade

def main():
    service = get_drive_service()
    trades = download_json(service)

    for i in range(len(trades)):
        trades[i] = update_trade(trades[i])

    upload_json(service, trades)

if __name__ == "__main__":
    main()
