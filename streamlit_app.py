# streamlit_app.py
import streamlit as st
from datetime import datetime, date
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
import json
import os
import io
from typing import List, Dict, Any, Optional

# ---------------------- GOOGLE DRIVE IMPORTS ----------------------
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ---------------------- APP SETUP ----------------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

DRIVE_FILENAME = "spread_trades.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
SERVICE_ACCOUNT_FILE = "service_account.json"  # fallback local file

# ---------------------- GOOGLE DRIVE CREDENTIALS ----------------------
def _get_credentials() -> Optional[service_account.Credentials]:
    """Load credentials from Streamlit secrets or local file."""
    # Streamlit secrets
    try:
        if st.secrets and "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            return creds
    except Exception:
        pass

    # Local fallback
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        try:
            creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            return creds
        except Exception:
            return None

    return None

@st.cache_resource
def get_drive_service():
    creds = _get_credentials()
    if creds is None:
        return None
    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception:
        return None

def find_file_id(service, filename: str) -> Optional[str]:
    if service is None:
        return None
    try:
        resp = service.files().list(q=f"name='{filename}' and trashed = false",
                                    spaces="drive",
                                    fields="files(id, name)").execute()
        files = resp.get("files", [])
        if not files:
            return None
        return files[0]["id"]
    except Exception:
        return None

def serialize_trades_for_storage(trades: List[Dict[str, Any]]) -> str:
    cleaned = []
    for t in trades:
        ct = {}
        for k, v in t.items():
            ct[k] = v.isoformat() if isinstance(v, date) else v
        cleaned.append(ct)
    return json.dumps(cleaned, indent=2)

def deserialize_trades_from_storage(raw: str) -> List[Dict[str, Any]]:
    try:
        loaded = json.loads(raw)
        out = []
        for t in loaded:
            nt = {}
            for k, v in t.items():
                if k in ("expiration", "entry_date") and isinstance(v, str):
                    try:
                        nt[k] = datetime.fromisoformat(v).date()
                    except Exception:
                        nt[k] = v
                else:
                    nt[k] = v
            out.append(nt)
        return out
    except Exception:
        return []

def save_to_drive(trades: List[Dict[str, Any]]) -> bool:
    service = get_drive_service()
    if service is None:
        st.warning("Google Drive credentials not found. Place service_account.json in app folder or add gcp_service_account to Streamlit secrets.")
        return False

    json_str = serialize_trades_for_storage(trades)
    fh = io.BytesIO(json_str.encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)

    existing_id = find_file_id(service, DRIVE_FILENAME)
    try:
        if existing_id:
            service.files().delete(fileId=existing_id).execute()
    except Exception:
        pass

    try:
        service.files().create(body={"name": DRIVE_FILENAME}, media_body=media, fields="id").execute()
        return True
    except Exception as e:
        st.error(f"Failed to save to Drive: {e}")
        return False

def load_from_drive() -> List[Dict[str, Any]]:
    service = get_drive_service()
    if service is None:
        st.info("Google Drive credentials not found; starting with empty trades.")
        return []

    file_id = find_file_id(service, DRIVE_FILENAME)
    if not file_id:
        return []

    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        raw = fh.read().decode("utf-8")
        trades = deserialize_trades_from_storage(raw)
        return trades
    except Exception:
        return []

# ---------------------- HELPER FUNCTIONS ----------------------
def days_to_expiry(expiry_date: date) -> int:
    return max((expiry_date - date.today()).days, 0)

def compute_derived(trade: dict) -> dict:
    short = float(trade["short_strike"])
    long = float(trade["long_strike"])
    credit = float(trade.get("credit", 0) or 0)
    width = abs(long - short)
    max_gain = credit
    max_loss = max(width - credit, 0)
    breakeven = short + credit
    dte = days_to_expiry(trade["expiration"])
    return {
        "width": width,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "dte": dte
    }

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

@st.cache_data(ttl=60)
def get_price(ticker: str):
    try:
        data = yf.Ticker(ticker).fast_info
        return float(data["last_price"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiration: str):
    try:
        ticker_obj = yf.Ticker(ticker)
        opt_chain = ticker_obj.option_chain(expiration)
        return opt_chain.calls, opt_chain.puts
    except Exception:
        return None, None

def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1 if option_type.lower() == 'put' else norm.cdf(d1)

def get_leg_data(ticker: str, expiration: date, strike: float, option_type='put'):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None, None
    leg_row = puts[puts['strike'] == strike]
    if leg_row.empty:
        return None, None
    price = leg_row['lastPrice'].values[0] if 'lastPrice' in leg_row.columns else None
    iv = leg_row['impliedVolatility'].values[0] * 100 if 'impliedVolatility' in leg_row.columns else None
    return price, iv

def get_short_leg_data(trade: dict):
    short_price, iv = get_leg_data(trade["ticker"], trade["expiration"], float(trade["short_strike"]), 'put')
    current_price = get_price(trade['ticker'])
    delta = None
    if current_price and iv:
        T = days_to_expiry(trade["expiration"]) / 365
        sigma = iv / 100
        r = 0.05
        delta = bsm_delta('put', current_price, float(trade["short_strike"]), T, r, sigma)
    return delta, iv, short_price

def get_long_leg_data(trade: dict):
    long_price, _ = get_leg_data(trade["ticker"], trade["expiration"], float(trade["long_strike"]), 'put')
    return long_price

def compute_spread_value(short_option_price, long_option_price, width, credit):
    if short_option_price is None or long_option_price is None or width - credit <= 0:
        return None
    spread_mark = short_option_price - long_option_price
    max_loss = width - credit
    return (spread_mark / max_loss) * 100

def compute_current_profit(short_price, long_price, credit, width):
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    current_profit = credit - spread_value
    return max(0, min((current_profit / credit) * 100, 100))

# ---------------------- FETCH SHORT IV ----------------------
def fetch_short_iv(ticker, short_strike, expiration):
    """Return implied volatility (%) of short leg, or None if not found"""
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    return float(short_row['impliedVolatility'].values[0] * 100)

# ---------------------- SESSION STATE ----------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = load_from_drive()
init_state()

# ---------------------- FORM & DASHBOARD ----------------------
# (Here you would copy your existing Streamlit form, save/load buttons, and dashboard code)
# Make sure the form calls `fetch_short_iv()` when adding a new trade.
# Use `save_to_drive()` and `load_from_drive()` as above.

