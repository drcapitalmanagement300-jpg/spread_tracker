# app.py (patched fully)
import streamlit as st
from datetime import date, datetime, timedelta
import json
import io
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt

# Persistence (Google OAuth + Drive)
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ----------------------------- IV30 Percent Rank helpers -----------------------------
ELEVATED_COLOR = "green"
HYPER_ELEVATED_COLOR = "#66ff66"
LOW_COLOR = "red"
HYPER_LOW_COLOR = "#ff6666"

@st.cache_data(ttl=60 * 30)
def get_price_simple(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        return float(fi["last_price"])
    except Exception:
        try:
            hist = tk.history(period="5d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return None

@st.cache_data(ttl=60 * 30)
def get_iv30_via_options(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        opts = tk.options
        if not opts:
            return None
        spot = get_price_simple(ticker)
        if spot is None:
            return None

        best_exp = None
        best_diff = None
        today = date.today()
        for exp_str in opts:
            try:
                exp_date = datetime.fromisoformat(exp_str).date()
            except Exception:
                try:
                    exp_date = pd.to_datetime(exp_str).date()
                except Exception:
                    continue
            dte = (exp_date - today).days
            diff = abs(dte - 30)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_exp = exp_str

        if best_exp is None:
            return None

        try:
            opt_chain = tk.option_chain(best_exp)
            calls, puts = opt_chain.calls, opt_chain.puts
        except Exception:
            return None

        if (calls is None or calls.empty) and (puts is None or puts.empty):
            return None

        low_strike = spot * 0.98
        high_strike = spot * 1.02

        ivs = []
        if calls is not None and not calls.empty:
            atm_calls = calls[(calls["strike"] >= low_strike) & (calls["strike"] <= high_strike)]
            if atm_calls is None or atm_calls.empty:
                atm_calls = calls.iloc[max(0, len(calls)//2 - 2): min(len(calls), len(calls)//2 + 3)]
            if 'impliedVolatility' in atm_calls.columns:
                ivs += [float(x) * 100 for x in atm_calls['impliedVolatility'].dropna().values]

        if puts is not None and not puts.empty:
            atm_puts = puts[(puts["strike"] >= low_strike) & (puts["strike"] <= high_strike)]
            if atm_puts is None or atm_puts.empty:
                atm_puts = puts.iloc[max(0, len(puts)//2 - 2): min(len(puts), len(puts)//2 + 3)]
            if 'impliedVolatility' in atm_puts.columns:
                ivs += [float(x) * 100 for x in atm_puts['impliedVolatility'].dropna().values]

        if not ivs:
            return None

        return float(np.mean(ivs))
    except Exception:
        return None

@st.cache_data(ttl=60 * 60 * 6)
def get_iv30_percent_rank(ticker: str) -> Optional[float]:
    try:
        current_iv30 = get_iv30_via_options(ticker)

        tk = yf.Ticker(ticker)
        prices = tk.history(period="1y", interval="1d")["Close"].dropna()
        if prices.empty:
            if current_iv30 is not None:
                return None
            return None

        returns = np.log(prices / prices.shift(1)).dropna()
        rolling30 = returns.rolling(window=30).std() * np.sqrt(252)
        rolling30_pct = (rolling30 * 100).dropna()

        if current_iv30 is None:
            if not rolling30_pct.empty:
                current_iv30 = float(rolling30_pct.iloc[-1])
            else:
                return None

        if rolling30_pct.empty:
            return None

        rank = float((rolling30_pct < current_iv30).sum() / len(rolling30_pct) * 100)
        return round(rank, 1)
    except Exception:
        return None

def iv_rank_status(rank: Optional[float]) -> str:
    if rank is None:
        return "N/A"
    if rank >= 70:
        return "Hyper Elevated"
    if rank >= 50:
        return "Elevated"
    if rank >= 30:
        return "Low"
    return "Hyper Low"

def iv_rank_color(rank: Optional[float]) -> str:
    if rank is None:
        return "black"
    if rank >= 70:
        return HYPER_ELEVATED_COLOR
    if rank >= 50:
        return ELEVATED_COLOR
    if rank >= 30:
        return LOW_COLOR
    return HYPER_LOW_COLOR

# ----------------------------- App core (UI & logic) -----------------------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth not available. Drive persistence disabled.")

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

_, logout_col = st.columns([9, 1])
with logout_col:
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
            st.success("Logged out (local). Reload to sign in.")
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ------------------- Helpers -------------------
def init_state():
    if "trades" not in st.session_state:
        loaded = []
        if drive_service:
            try:
                loaded = load_from_drive(drive_service) or []
            except Exception:
                loaded = []
        st.session_state.trades = loaded if loaded else []

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
    return {"width": width, "max_gain": max_gain, "max_loss": max_loss, "breakeven": breakeven, "dte": dte}

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

# ------------------- Initialize session & UI -------------------
init_state()

with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        ticker = st.text_input("Ticker (e.g. AAPL)").upper()
        short_strike = st.number_input("Short strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long strike", min_value=0.0, format="%.2f")
        # IV30 rank display
        if ticker:
            iv30_rank = get_iv30_percent_rank(ticker)
            iv30_status = iv_rank_status(iv30_rank)
            iv30_color = iv_rank_color(iv30_rank)
            if iv30_rank is None:
                st.markdown("<div>IV30 Percent Rank: <strong>N/A</strong> — <em>N/A</em></div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='color:{iv30_color}'>IV30 Percent Rank: <strong>{iv30_rank:.1f}%</strong></div>",
                    unsafe_allow_html=True
                )

    with col2:
        expiration = st.date_input("Expiration date", value=date.today())
        credit = st.number_input("Credit received (per share)", min_value=0.0, format="%.2f")
    with col3:
        entry_date = st.date_input("Entry date", value=date.today())
        notes = st.text_input("Notes (optional)")

    submitted = st.form_submit_button("Add trade for monitoring")

    if submitted:
        if not ticker:
            st.warning("Please provide a ticker symbol.")
        elif long_strike >= short_strike:
            st.warning("Long strike must be LOWER than short strike for put credit spreads.")
        else:
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv30_rank": iv30_rank,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)
            saved_to_drive = False
            if drive_service:
                try:
                    saved_to_drive = save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    saved_to_drive = False

            if saved_to_drive:
                st.success(f"Added {ticker} — saved to Drive.")
            else:
                st.success(f"Added {ticker} locally. (Drive not configured or save failed)")

st.markdown("---")
st.subheader("Active Trades")
if not st.session_state.trades:
    st.info("No trades added yet. Use the form above to add your first spread.")
