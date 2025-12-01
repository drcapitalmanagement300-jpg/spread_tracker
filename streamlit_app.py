# app.py
import streamlit as st
from datetime import date, datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import json
import io

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# =========================================================
#                GOOGLE DRIVE HELPERS (NEW)
# =========================================================

@st.cache_resource
def init_drive():
    """Authenticate using Streamlit secrets (service account)."""
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        st.error(f"Failed Google Drive auth: {e}")
        return None


DRIVE_FILE_NAME = "credit_spreads.json"


def find_file(service):
    """Return file_id if file exists."""
    try:
        query = f"name='{DRIVE_FILE_NAME}'"
        results = service.files().list(q=query, fields="files(id)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None
    except:
        return None


def load_from_drive(service):
    """Load JSON data from Drive file."""
    try:
        file_id = find_file(service)
        if not file_id:
            return []

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        fh.seek(0)
        return json.loads(fh.read().decode())
    except:
        return []


def save_to_drive(service, data):
    """Upload JSON to Google Drive."""
    file_id = find_file(service)

    fh = io.BytesIO(json.dumps(data, indent=2).encode())
    media = MediaIoBaseUpload(fh, mimetype="application/json")

    if file_id:
        service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
    else:
        file_metadata = {"name": DRIVE_FILE_NAME}
        service.files().create(
            body=file_metadata,
            media_body=media
        ).execute()


drive = init_drive()



# =========================================================
#               ORIGINAL HELPERS (UNCHANGED)
# =========================================================

def init_state():
    if "trades" not in st.session_state:
        if drive:
            st.session_state.trades = load_from_drive(drive)
        else:
            st.session_state.trades = []


def days_to_expiry(exp):
    return max((exp - date.today()).days, 0)


def compute_derived(t):
    short = float(t["short_strike"])
    long = float(t["long_strike"])
    credit = float(t["credit"])
    width = abs(long - short)
    max_gain = credit
    max_loss = width - credit
    return {
        "width": width,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakeven": short + credit,
        "dte": days_to_expiry(t["expiration"])
    }


def format_money(x):
    try:
        return f"${float(x):.2f}"
    except:
        return "-"


@st.cache_data(ttl=60)
def get_price(t):
    try:
        return float(yf.Ticker(t).fast_info["last_price"])
    except:
        return None


@st.cache_data(ttl=60)
def get_option_chain(t, exp_str):
    try:
        opt = yf.Ticker(t).option_chain(exp_str)
        return opt.calls, opt.puts
    except:
        return None, None


def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1 if option_type == "put" else norm.cdf(d1)


def get_leg_data(ticker, exp, strike):
    _, puts = get_option_chain(ticker, exp.isoformat())
    if puts is None or puts.empty:
        return None, None
    row = puts[puts["strike"] == strike]
    if row.empty:
        return None, None
    return row["lastPrice"].values[0], row["impliedVolatility"].values[0] * 100


def get_short_leg_data(t):
    price, iv = get_leg_data(t["ticker"], t["expiration"], float(t["short_strike"]))
    delta = None
    S = get_price(t["ticker"])
    if S and iv:
        T = days_to_expiry(t["expiration"]) / 365
        delta = bsm_delta("put", S, float(t["short_strike"]), T, 0.05, iv/100)
    return delta, iv, price


def get_long_leg_data(t):
    price, _ = get_leg_data(t["ticker"], t["expiration"], float(t["long_strike"]))
    return price


def compute_spread_value(short_p, long_p, width, credit):
    if short_p is None or long_p is None:
        return None
    return ((short_p - long_p) / (width - credit)) * 100


def compute_current_profit(short_p, long_p, credit, width):
    if short_p is None or long_p is None:
        return None
    sp = short_p - long_p
    profit = credit - sp
    return max(0, min((profit / credit) * 100, 100))


def fetch_short_iv(t, strike, exp):
    _, puts = get_option_chain(t, exp.isoformat())
    if puts is None or puts.empty:
        return None
    row = puts[puts["strike"] == strike]
    if row.empty:
        return None
    return row["impliedVolatility"].values[0] * 100


def evaluate_rules(t, derived, S, delta, iv, short_p, long_p):
    v = {"other_rules": False, "iv_rule": False}

    if delta and abs(delta) >= 0.40:
        v["other_rules"] = True

    sp = compute_spread_value(short_p, long_p, derived["width"], t["credit"])
    if sp and sp >= 150:
        v["other_rules"] = True

    if derived["dte"] <= 7:
        v["other_rules"] = True

    if t["entry_iv"] and iv and iv > t["entry_iv"]:
        v["iv_rule"] = True

    return v, abs(delta) if delta else None, sp



# =========================================================
#                   LOAD STATE
# =========================================================

init_state()


# =========================================================
#                     INPUT FORM
# =========================================================
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")

    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.text_input("Ticker").upper()
        short_strike = st.number_input("Short strike", min_value=0.0)
        long_strike = st.number_input("Long strike", min_value=0.0)

    with col2:
        expiration = st.date_input("Expiration", value=date.today())
        credit = st.number_input("Credit received", min_value=0.0)

    with col3:
        entry_date = st.date_input("Entry date", value=date.today())
        notes = st.text_input("Notes")

    submitted = st.form_submit_button("Add trade for monitoring")

    if submitted:
        if not ticker:
            st.warning("Ticker required.")
        elif long_strike >= short_strike:
            st.warning("Long strike must be LOWER than short strike.")
        else:
            iv = fetch_short_iv(ticker, short_strike, expiration)
            t = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv": iv,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(t)

            if drive:
                save_to_drive(drive, st.session_state.trades)

            st.success(f"Added {ticker}. Entry IV: {iv}")


st.markdown("---")


# =========================================================
#                ACTIVE TRADES DISPLAY
# =========================================================

st.subheader("Active Trades")

if not st.session_state.trades:
    st.info("No trades yet.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
        S = get_price(t["ticker"])
        delta, iv, sp = get_short_leg_data(t)
        long_p = get_long_leg_data(t)
        current_profit = compute_current_profit(sp, long_p, t["credit"], derived["width"])

        rules, abs_delta, spread_val = evaluate_rules(
            t, derived, S, delta, iv, sp, long_p
        )

        cols = st.columns(2)

        with cols[0]:
            st.markdown(
                f"""
                **Ticker:** {t['ticker']}  
                **Underlying:** {S}  
                **Short Strike:** {t['short_strike']}  
                **Long Strike:** {t['long_strike']}  
                **Width:** {derived['width']}  
                **Expiration:** {t['expiration']}  
                **DTE:** {derived['dte']}  
                **Max Gain:** {format_money(derived['max_gain'])}  
                **Max Loss:** {format_money(derived['max_loss'])}  
                """
            )

        with cols[1]:
            st.markdown(
                f"""
                **Delta:** {abs_delta}  
                **Spread Value:** {spread_val}%  
                **Current Profit:** {current_profit}%  
                **Entry IV:** {t['entry_iv']}%  
                **Current IV:** {iv}%  
                """
            )

        if st.button("Remove", key=f"rm_{i}"):
            st.session_state.trades.pop(i)
            if drive:
                save_to_drive(drive, st.session_state.trades)
            st.experimental_rerun()

st.caption("Synced with Google Drive — no pydrive2 required.")
