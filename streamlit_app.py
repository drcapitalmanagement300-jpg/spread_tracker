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

# Google Drive API imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ---------------------- DRIVE CONFIG ----------------------
DRIVE_FILENAME = "spread_trades.json"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
SERVICE_ACCOUNT_FILE = "service_account.json"  # local fallback

# ---------------------- DRIVE HELPERS ----------------------
def _get_credentials() -> Optional[service_account.Credentials]:
    """
    Obtain service account credentials from:
      1) Streamlit secrets (recommended for deployed apps)
      2) Local file (for development)
    Returns a Credentials object or None if neither is available.
    """
    # 1) Try Streamlit secrets
    try:
        if st.secrets and "gcp_service_account" in st.secrets:
            info = st.secrets["gcp_service_account"]
            if not isinstance(info, dict):
                st.error("Streamlit secret 'gcp_service_account' is not a dict! Check TOML formatting.")
                return None
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            st.write("✅ Loaded Google Drive credentials from Streamlit secrets.")
            return creds
    except Exception as e:
        st.error(f"Failed to load credentials from Streamlit secrets: {e}")

    # 2) Try local file fallback
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        try:
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            st.write(f"✅ Loaded Google Drive credentials from local file {SERVICE_ACCOUNT_FILE}")
            return creds
        except Exception as e:
            st.error(f"Failed to load credentials from local file: {e}")

    # No credentials found
    st.warning(
        "⚠️ Google Drive credentials not found. Add a service_account.json file locally "
        "or configure Streamlit secrets."
    )
    return None

@st.cache_resource
def get_drive_service():
    """
    Returns an authorized Drive v3 service object, or None if credentials missing.
    Cached with st.cache_resource so we reuse the client across reruns.
    """
    creds = _get_credentials()
    if creds is None:
        return None
    try:
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        st.error(f"Failed to build Google Drive service: {e}")
        return None

def find_file_id(service, filename: str) -> Optional[str]:
    """Return file id for the given filename in Drive, or None if not found."""
    if service is None:
        return None
    try:
        resp = service.files().list(
            q=f"name='{filename}' and trashed = false",
            spaces="drive",
            fields="files(id, name)"
        ).execute()
        files = resp.get("files", [])
        if not files:
            return None
        return files[0]["id"]
    except Exception:
        return None

def serialize_trades_for_storage(trades: List[Dict[str, Any]]) -> str:
    """Convert Python objects (dates) into JSON-friendly types and return JSON string."""
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

def deserialize_trades_from_storage(raw: str) -> List[Dict[str, Any]]:
    """Parse JSON and convert ISO date strings back to date objects where appropriate."""
    try:
        loaded = json.loads(raw)
        out = []
        for t in loaded:
            nt = {}
            for k, v in t.items():
                if isinstance(v, str):
                    if k in ("expiration", "entry_date", "created_at"):
                        try:
                            dt = datetime.fromisoformat(v)
                            if k == "created_at":
                                nt[k] = v
                            else:
                                nt[k] = dt.date()
                        except Exception:
                            nt[k] = v
                    else:
                        nt[k] = v
                else:
                    nt[k] = v
            out.append(nt)
        return out
    except Exception:
        return []

def save_to_drive(trades: List[Dict[str, Any]]) -> bool:
    """Save trades list as JSON file to Google Drive (overwrites existing file with same name)."""
    service = get_drive_service()
    if service is None:
        st.warning("Google Drive credentials not found. Add service_account.json or Streamlit secrets.")
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

    file_metadata = {"name": DRIVE_FILENAME}
    try:
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        return True
    except Exception as e:
        st.error(f"Failed to save to Drive: {e}")
        return False

def load_from_drive() -> List[Dict[str, Any]]:
    """Load trades JSON from Google Drive and return list of trades (with date fields converted)."""
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

# ---------------------- SESSION STATE ----------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = load_from_drive()
init_state()

# ---------------------- UTILITY FUNCTIONS ----------------------
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

# ---------------------- Remaining Option/Spread Helpers ----------------------
# (You can keep all your get_option_chain, bsm_delta, get_leg_data, get_short_leg_data, get_long_leg_data,
# compute_spread_value, compute_current_profit, fetch_short_iv, evaluate_rules functions exactly as before)

# ---------------------- TRADE INPUT FORM ----------------------
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

st.markdown("---")

# ---------------------- SAVE / LOAD BUTTONS ----------------------
colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save trades to Google Drive"):
        ok = save_to_drive(st.session_state.trades)
        if ok:
            st.success("Saved to Google Drive.")
        else:
            st.error("Save failed. Check credentials and try again.")
with colB:
    if st.button("📥 Load trades from Google Drive"):
        loaded = load_from_drive()
        if loaded:
            st.session_state.trades = loaded
            st.success("Loaded from Google Drive.")
            st.experimental_rerun()
        else:
            st.info("No saved trades found or credentials missing.")

# ---------------------- ACTIVE TRADES DASHBOARD ----------------------
# Keep your dashboard code exactly as it was
