# app.py
import streamlit as st
from datetime import datetime, date
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
import json
import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ------------------- Google Drive Config -------------------
DRIVE_FILENAME = "spread_trades.json"
SERVICE_ACCOUNT_FILE = "service_account.json"  # local fallback
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def _get_credentials():
    """Get credentials from Streamlit secrets or local JSON."""
    # 1) Streamlit secrets
    try:
        if st.secrets and "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            return creds
    except Exception:
        pass

    # 2) Local file fallback
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

def find_file_id(service, filename):
    if service is None:
        return None
    try:
        resp = service.files().list(
            q=f"name='{filename}' and trashed=false",
            spaces="drive",
            fields="files(id,name)"
        ).execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None

def serialize_trades(trades):
    cleaned = []
    for t in trades:
        ct = {}
        for k, v in t.items():
            if isinstance(v, date):
                ct[k] = v.isoformat()
            else:
                ct[k] = v
        cleaned.append(ct)
    return json.dumps(cleaned, indent=2)

def deserialize_trades(raw):
    try:
        loaded = json.loads(raw)
        out = []
        for t in loaded:
            nt = {}
            for k, v in t.items():
                if isinstance(v, str) and k in ("expiration", "entry_date"):
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

def save_to_drive(trades):
    service = get_drive_service()
    if service is None:
        st.warning("Google Drive credentials not found. Place service_account.json in app folder or add gcp_service_account to Streamlit secrets.")
        return False
    json_str = serialize_trades(trades)
    fh = io.BytesIO(json_str.encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)
    existing_id = find_file_id(service, DRIVE_FILENAME)
    try:
        if existing_id:
            service.files().delete(fileId=existing_id).execute()
    except Exception:
        pass
    try:
        service.files().create(body={"name":DRIVE_FILENAME}, media_body=media, fields="id").execute()
        return True
    except Exception as e:
        st.error(f"Failed to save to Drive: {e}")
        return False

def load_from_drive():
    service = get_drive_service()
    if service is None:
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
            _, done = downloader.next_chunk()
        fh.seek(0)
        raw = fh.read().decode("utf-8")
        return deserialize_trades(raw)
    except Exception:
        return []

# ------------------- Helpers (your existing code) ---------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = load_from_drive()

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
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    elif option_type.lower() == 'put':
        return norm.cdf(d1)-1
    else:
        return None

def get_leg_data(ticker: str, expiration: date, strike: float, option_type='put'):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None, None
    leg_row = puts[puts['strike'] == strike]
    if leg_row.empty:
        return None, None
    price = leg_row['lastPrice'].values[0] if 'lastPrice' in leg_row.columns else None
    iv = leg_row['impliedVolatility'].values[0]*100 if 'impliedVolatility' in leg_row.columns else None
    return price, iv

def get_short_leg_data(trade: dict):
    short_price, iv = get_leg_data(trade["ticker"], trade["expiration"], float(trade["short_strike"]), 'put')
    current_price = get_price(trade['ticker'])
    delta = None
    if current_price and iv:
        T = days_to_expiry(trade["expiration"])/365
        sigma = iv/100
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
    return (spread_mark/max_loss)*100

def compute_current_profit(short_price, long_price, credit, width):
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    current_profit = credit - spread_value
    return max(0, min((current_profit/credit)*100, 100))

def fetch_short_iv(ticker, short_strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike']==short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    return short_row['impliedVolatility'].values[0]*100

# ------------------- Initialize -------------------
init_state()

# ------------------- Trade Input Form -------------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        ticker = st.text_input("Ticker (e.g. AAPL)").upper()
        short_strike = st.number_input("Short strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long strike", min_value=0.0, format="%.2f")
    with col2:
        expiration = st.date_input("Expiration date", value=date.today())
        credit = st.number_input("Credit received (per share)", min_value=0.0, format="%.2f")
    with col3:
        entry_date = st.date_input("Entry date", value=date.today())
        notes = st.text_input("Notes (optional)")
        st.write("")

    submitted = st.form_submit_button("Add trade for monitoring")
    if submitted:
        if not ticker:
            st.warning("Please provide a ticker symbol.")
        elif long_strike >= short_strike:
            st.warning("For a put credit spread, long strike should be LOWER than short strike.")
        else:
            auto_iv = fetch_short_iv(ticker, short_strike, expiration)
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv": auto_iv,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)
            st.success(f"Added {ticker} {short_strike}/{long_strike} exp {expiration} | Entry IV: {auto_iv if auto_iv else 'N/A'}")

# ------------------- Save / Load Buttons -------------------
st.markdown("---")
colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save trades to Google Drive"):
        ok = save_to_drive(st.session_state.trades)
        if ok:
            st.success("Saved to Google Drive.")
        else:
            st.error("Save failed. Check credentials.")

with colB:
    if st.button("📥 Load trades from Google Drive"):
        loaded = load_from_drive()
        if loaded:
            st.session_state.trades = loaded
            st.success("Loaded from Google Drive.")
            st.experimental_rerun()
        else:
            st.info("No saved trades found or credentials missing.")

# ------------------- Active Trades Dashboard -------------------
# (Everything from your previous code for cards, charts, rules remains unchanged)
# ... include the full dashboard rendering logic here ...
