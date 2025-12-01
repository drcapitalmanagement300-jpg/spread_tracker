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

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")


# =========================================================
#                  GOOGLE DRIVE HELPERS
# =========================================================

@st.cache_resource
def init_drive():
    """Authenticate Google Drive using Streamlit secrets."""
    try:
        creds_dict = st.secrets["gcp_service_account"]

        scope = [
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

        gauth = GoogleAuth()
        gauth.credentials = credentials
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        st.error(f"Drive init failed: {e}")
        return None


DRIVE_FILE_NAME = "credit_spreads.json"


def load_from_drive(drive):
    """Load JSON file from Google Drive."""
    try:
        file_list = drive.ListFile(
            {"q": f"title='{DRIVE_FILE_NAME}'"}
        ).GetList()

        if not file_list:
            return []

        file = file_list[0]
        content = file.GetContentString()
        return json.loads(content)
    except:
        return []


def save_to_drive(drive, data):
    """Save JSON to Google Drive."""
    file_list = drive.ListFile(
        {"q": f"title='{DRIVE_FILE_NAME}'"}
    ).GetList()

    # If file exists — update it
    if file_list:
        file = file_list[0]
    else:
        file = drive.CreateFile({"title": DRIVE_FILE_NAME})

    file.SetContentString(json.dumps(data, indent=2))
    file.Upload()


# Initialize Google Drive
drive = init_drive()


# =========================================================
#               ORIGINAL APP HELPERS (UNCHANGED)
# =========================================================

def init_state():
    if "trades" not in st.session_state:
        if drive:
            st.session_state.trades = load_from_drive(drive)
        else:
            st.session_state.trades = []


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
    except:
        return "-"


@st.cache_data(ttl=60)
def get_price(ticker: str):
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except:
        return None


@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiration: str):
    try:
        opt = yf.Ticker(ticker).option_chain(expiration)
        return opt.calls, opt.puts
    except:
        return None, None


def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def get_leg_data(ticker: str, expiration: date, strike: float):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None, None
    row = puts[puts['strike'] == strike]
    if row.empty:
        return None, None
    return (
        row['lastPrice'].values[0],
        row['impliedVolatility'].values[0] * 100
    )


def get_short_leg_data(trade: dict):
    short_price, iv = get_leg_data(
        trade["ticker"], trade["expiration"], float(trade["short_strike"])
    )
    current_price = get_price(trade["ticker"])
    delta = None
    if current_price and iv:
        T = days_to_expiry(trade["expiration"]) / 365
        r = 0.05
        delta = bsm_delta('put', current_price, float(trade["short_strike"]), T, r, iv / 100)
    return delta, iv, short_price


def get_long_leg_data(trade: dict):
    price, _ = get_leg_data(
        trade["ticker"], trade["expiration"], float(trade["long_strike"])
    )
    return price


def compute_spread_value(short_p, long_p, width, credit):
    if short_p is None or long_p is None:
        return None
    return ((short_p - long_p) / (width - credit)) * 100


def compute_current_profit(short_p, long_p, credit, width):
    if short_p is None or long_p is None or credit <= 0:
        return None
    spread_value = short_p - long_p
    current_profit = credit - spread_value
    return max(0, min((current_profit / credit) * 100, 100))


def fetch_short_iv(ticker, strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    row = puts[puts["strike"] == strike]
    if row.empty:
        return None
    return row["impliedVolatility"].values[0] * 100


def evaluate_rules(trade, derived, current_price, delta, current_iv, short_p, long_p):
    rule_violations = {"other_rules": False, "iv_rule": False}

    # delta rule
    if delta is not None and abs(delta) >= 0.40:
        rule_violations["other_rules"] = True

    # spread value rule
    sp = compute_spread_value(short_p, long_p, derived["width"], trade["credit"])
    if sp is not None and sp >= 150:
        rule_violations["other_rules"] = True

    # dte rule
    if derived["dte"] <= 7:
        rule_violations["other_rules"] = True

    # IV rule
    if trade.get("entry_iv") and current_iv and current_iv > trade["entry_iv"]:
        rule_violations["iv_rule"] = True

    return rule_violations, abs(delta) if delta else None, sp


# =========================================================
#                 LOAD INITIAL STATE
# =========================================================
init_state()


# =========================================================
#                INPUT FORM (UNCHANGED)
# =========================================================
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")
    col1, col2, col3 = st.columns([2, 2, 2])

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
            entry_iv = fetch_short_iv(ticker, short_strike, expiration)
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv": entry_iv,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)

            # SAVE TO GOOGLE DRIVE
            if drive:
                save_to_drive(drive, st.session_state.trades)

            st.success(f"Added {ticker} spread. Entry IV: {entry_iv}")


st.markdown("---")


# =========================================================
#           ACTIVE TRADES (UNCHANGED UI)
# =========================================================
st.subheader("Active Trades")

if not st.session_state.trades:
    st.info("No trades yet.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
        current_price = get_price(t["ticker"])
        delta, current_iv, short_price = get_short_leg_data(t)
        long_price = get_long_leg_data(t)

        current_profit = compute_current_profit(
            short_price, long_price, t["credit"], derived["width"]
        )

        rules, abs_delta, spread_val = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_price, long_price
        )

        # Card layout
        cols = st.columns([3, 3])

        with cols[0]:
            st.markdown(
                f"""
                **Ticker:** {t['ticker']}  
                **Underlying:** {current_price}  
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
                **Current IV:** {current_iv}%  
                """
            )

        # Remove button
        if st.button("Remove", key=f"rm_{i}"):
            st.session_state.trades.pop(i)
            if drive:
                save_to_drive(drive, st.session_state.trades)
            st.experimental_rerun()

st.caption("Powered by live option chain data + Google Drive autosync.")
