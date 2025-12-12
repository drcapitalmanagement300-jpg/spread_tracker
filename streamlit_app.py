# app.py (patched with auto-refresh every 10 minutes)
import streamlit as st
from datetime import date, datetime
import json
import io
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh  # <- auto-refresh import

# Persistence (Google OAuth + Drive) — provided in persistence.py
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ------------------- Auto-refresh every 10 minutes -------------------
# Interval is in milliseconds: 10 minutes = 600,000 ms
count = st_autorefresh(interval=600_000, key="data_refresh")

# ----------------------------- App core (UI & logic) -----------------------------
# Ensure the user is logged-in via OAuth
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth not fully configured. Running locally.")

# Build Drive service (may be None if not signed in)
drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# small logout button in header area
_, logout_col = st.columns([9, 1])
with logout_col:
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
            st.success("Logged out (local). Reload the page to sign in again.")
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

        # Ensure all loaded trades have pnl_history (backfill if missing)
        for tr in loaded:
            if "pnl_history" not in tr:
                tr["pnl_history"] = []

        st.session_state.trades = loaded if loaded else []

def days_to_expiry(expiry_date: date) -> int:
    # Accept either date or ISO string
    if isinstance(expiry_date, str):
        try:
            expiry_date = date.fromisoformat(expiry_date)
        except Exception:
            # fallback: return 0
            return 0
    return max((expiry_date - date.today()).days, 0)

def compute_derived(trade: dict) -> dict:
    # Ensure numeric conversion and handle string expiration
    short = float(trade["short_strike"])
    long = float(trade["long_strike"])
    credit = float(trade.get("credit", 0) or 0)
    width = abs(long - short)
    max_gain = credit
    max_loss = max(width - credit, 0)
    breakeven = short + credit

    # If expiration stored as string, parse
    expiry = trade["expiration"]
    if isinstance(expiry, str):
        try:
            expiry = date.fromisoformat(expiry)
        except Exception:
            # if parsing fails, fallback to today (0 DTE)
            expiry = date.today()

    dte = days_to_expiry(expiry)
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

# ------------------- Cached price / options functions -------------------
@st.cache_data(ttl=600)  # cache for 10 minutes
def get_price(ticker: str):
    try:
        data = yf.Ticker(ticker).fast_info
        return float(data["last_price"])
    except Exception:
        return None

@st.cache_data(ttl=600)  # cache for 10 minutes
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    elif option_type.lower() == 'put':
        return norm.cdf(d1) - 1
    else:
        return None

def get_leg_data(ticker: str, expiration: date, strike: float, option_type='put'):
    # Accept expiration as date or ISO string
    exp_str = expiration.isoformat() if isinstance(expiration, date) else str(expiration)
    _, puts = get_option_chain(ticker, exp_str)
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

def fetch_short_iv(ticker, short_strike, expiration):
    exp_str = expiration.isoformat() if isinstance(expiration, date) else str(expiration)
    _, puts = get_option_chain(ticker, exp_str)
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    iv = short_row['impliedVolatility'].values[0] * 100
    return iv

def evaluate_rules(trade, derived, current_price, delta, current_iv, short_option_price, long_option_price):
    rule_violations = {"other_rules": False, "iv_rule": False}
    abs_delta = abs(delta) if delta is not None else None
    if abs_delta is not None and abs_delta >= 0.40:
        rule_violations["other_rules"] = True
    spread_value_percent = compute_spread_value(short_option_price, long_option_price, derived["width"], trade["credit"])
    if spread_value_percent is not None and spread_value_percent >= 150:
        rule_violations["other_rules"] = True
    if derived["dte"] <= 7:
        rule_violations["other_rules"] = True
    entry_iv = trade.get("entry_iv")
    if entry_iv and current_iv and current_iv > entry_iv:
        rule_violations["iv_rule"] = True
    return rule_violations, abs_delta, spread_value_percent

### NEW: helper to update pnl history ###
def update_pnl_history(trade: dict, current_profit_percent: Optional[float], dte: int) -> bool:
    """
    Append a new PnL point for today if it doesn't already exist.
    Returns True if history was modified.
    """
    today_iso = date.today().isoformat()

    # Initialize if missing
    if "pnl_history" not in trade:
        trade["pnl_history"] = []

    # If today's entry exists, do nothing
    if any(entry.get("date") == today_iso for entry in trade["pnl_history"]):
        return False

    # Append a new history item. Store profit as float or None.
    trade["pnl_history"].append({
        "date": today_iso,
        # store the DTE at time of recording as well for reference (not used for plotting)
        "dte": int(dte) if dte is not None else None,
        "profit": float(current_profit_percent) if current_profit_percent is not None else None
    })
    return True

# ----------------------------- Initialize session & UI -----------------------------
init_state()

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
                "created_at": datetime.utcnow().isoformat(),
                "pnl_history": []  # NEW: start with empty history
            }
            st.session_state.trades.append(trade)

            # Try to save to Drive; if Drive not configured, show success locally
            saved_to_drive = False
            if drive_service:
                try:
                    saved_to_drive = save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    saved_to_drive = False

            if saved_to_drive:
                st.success(f"Added {ticker} — saved to Drive. Entry IV: {auto_iv if auto_iv else 'N/A'}")
            else:
                st.success(f"Added {ticker} locally. (Drive not configured or save failed)")

st.markdown("---")

# ----------------------------- Display active trades -----------------------------
st.subheader("Active Trades")
if not st.session_state.trades:
    st.info("No trades added yet. Use the form above to add your first spread.")
else:
    # track if we added any pnl history entries so we can save once at the end
    history_changed = False

    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
        current_price = get_price(t['ticker'])
        delta, current_iv, short_option_price = get_short_leg_data(t)
        long_option_price = get_long_leg_data(t)
        current_profit_percent = compute_current_profit(short_option_price, long_option_price, t["credit"], derived["width"])
        rule_violations, abs_delta, spread_value_percent = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_option_price, long_option_price
        )

        # Update pnl history for today (if not already present)
        try:
            changed = update_pnl_history(t, current_profit_percent, derived["dte"])
            if changed:
                history_changed = True
        except Exception:
            # Fail silently so UI still renders
            pass

        abs_delta_str = f"{abs_delta:.2f}" if abs_delta is not None else "-"
        spread_value_str = f"{spread_value_percent:.0f}%" if spread_value_percent is not None else ""
        current_profit_str = f"{current_profit_percent:.1f}%" if current_profit_percent is not None else ""
        current_price_str = f"{current_price:.2f}" if current_price is not None else "-"

        if rule_violations["other_rules"]:
            status_icon = "❌"
            status_text = "Some critical rules are violated."
        else:
            status_icon = "✅"
            status_text = "All rules are satisfied."

        card_cols = st.columns([3,3])
        with card_cols[0]:
            st.markdown(
                f"""
<div style='background-color:rgba(0,100,0,0.1); padding:15px; border-radius:10px; height:100%'>
Ticker: {t['ticker']}  <br>
Underlying Price: {current_price_str}  <br>
Short Strike: {t['short_strike']}  <br>
Long Strike: {t['long_strike']}  <br>
Spread Width: {derived['width']}  <br>
Expiration Date: {t['expiration']}  <br>
Current DTE: {derived['dte']}  <br>
Max Gain: {format_money(derived['max_gain'])}  <br>
Max Loss: {format_money(derived['max_loss'])}  
</div>
""", unsafe_allow_html=True)

            st.markdown(f"<div style='margin-top:10px; font-size:20px'>{status_icon} {status_text}</div>", unsafe_allow_html=True)

        with card_cols[1]:
            delta_color = "red" if abs_delta is not None and abs_delta >= 0.40 else "green"
            spread_color = "red" if spread_value_percent is not None and spread_value_percent >= 150 else "green"
            dte_color = "red" if derived['dte'] <= 7 else "green"

            if current_profit_percent is None:
                profit_color = "black"
            elif current_profit_percent < 50:
                profit_color = "green"
            elif 50 <= current_profit_percent <= 75:
                profit_color = "yellow"
            else:
                profit_color = "red"

            if current_iv is None or t.get("entry_iv") is None:
                iv_color = "black"
            elif current_iv == t.get("entry_iv"):
                iv_color = "yellow"
            elif current_iv > t.get("entry_iv"):
                iv_color = "red"
            else:
                iv_color = "green"

            entry_iv_display = f"{t.get('entry_iv'):.1f}%" if isinstance(t.get("entry_iv"), (int, float)) else (str(t.get("entry_iv")) or "N/A")
            current_iv_display = f"{current_iv:.1f}%" if isinstance(current_iv, (int, float)) else (str(current_iv) or "N/A")

            st.markdown(
                f"""
Short Delta: <span style='color:{delta_color}'>{abs_delta_str}</span> | Must be less than or equal to 0.40 <br>
Spread Value: <span style='color:{spread_color}'>{spread_value_str}</span> | Must be less than or equal to 150% of credit <br>
DTE: <span style='color:{dte_color}'>{derived['dte']}</span> | Must be greater than 7 <br>
Current Profit: <span style='color:{profit_color}'>{current_profit_str}</span> | 50-75% Max profit target <br>
""", unsafe_allow_html=True)

            # ------------------ REAL PnL HISTORY CHART ------------------
            try:
                # If there's history use it; otherwise fallback to a single point (today)
                if t.get("pnl_history"):
                    pnl_df = pd.DataFrame(t["pnl_history"])
                    # Ensure date column is datetime
                    pnl_df["date"] = pd.to_datetime(pnl_df["date"])
                    # Sort oldest -> newest
                    pnl_df = pnl_df.sort_values("date")
                    # If profit column missing or null, fill with 0 for plotting (but keep NaNs handled)
                    # We'll keep NaNs as-is so gaps are visible; but Altair prefers numbers, so fill to 0 when all null
                    if "profit" not in pnl_df.columns or pnl_df["profit"].isnull().all():
                        pnl_df["profit"] = pnl_df.get("profit", pd.Series([None]*len(pnl_df))).fillna(0)
                else:
                    # Build a single-row DataFrame for today
                    pnl_df = pd.DataFrame({
                        "date": [pd.to_datetime(date.today().isoformat())],
                        "profit": [float(current_profit_percent) if current_profit_percent is not None else None]
                    })

                # Build the line chart
                chart = (
                    alt.Chart(pnl_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("profit:Q", title="Profit %", scale=alt.Scale(domain=[0,100]))
                    )
                    .properties(height=250)
                )

                # Add 50% and 75% horizontal lines
                line_50 = alt.Chart(pd.DataFrame({'y':[50]})).mark_rule(strokeDash=[5,5]).encode(y='y')
                line_75 = alt.Chart(pd.DataFrame({'y':[75]})).mark_rule(strokeDash=[5,5]).encode(y='y')

                final_chart = chart + line_50 + line_75
                st.altair_chart(final_chart, use_container_width=True)
            except Exception as e:
                # Fallback to a simple text if plotting fails
                st.write("PnL chart unavailable.")
                st.write(f"Error: {e}")

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            saved = False
            if drive_service:
                try:
                    saved = save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    saved = False

            if saved:
                st.success("Saved updated trades to Drive.")
            else:
                st.warning("Removed locally but failed to save to Drive (or Drive not configured).")
            try:
                st.experimental_rerun()
            except Exception:
                pass

    # If any pnl history entries were added during this run, try to persist once
    if history_changed and drive_service:
        try:
            save_to_drive(drive_service, st.session_state.trades)
        except Exception:
            # don't raise - persistence failure shouldn't break the UI
            pass

st.markdown("---")

# Manual Save/Load (uses persistence API)
colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save all trades to Google Drive now"):
        saved = False
        if drive_service:
            try:
                saved = save_to_drive(drive_service, st.session_state.trades)
            except Exception:
                saved = False
        if saved:
            st.success("Saved to Drive successfully.")
        else:
            st.error("Failed to save to Drive. Check logs or ensure you're signed in.")

with colB:
    if st.button("📥 Reload trades from Google Drive"):
        loaded = []
        if drive_service:
            try:
                loaded = load_from_drive(drive_service) or []
            except Exception:
                loaded = []
        if loaded:
            # ensure pnl_history backfill
            for tr in loaded:
                if "pnl_history" not in tr:
                    tr["pnl_history"] = []
            st.session_state.trades = loaded
            st.success("Loaded trades from Drive.")
            try:
                st.experimental_rerun()
            except Exception:
                pass
        else:
            st.info("No trades found on Drive (or load failed).")

# ------------------- Countdown until next refresh -------------------
import time

refresh_interval_sec = 600  # 10 minutes

# This container will hold the countdown text
countdown_placeholder = st.empty()

# Calculate time since last refresh
start_time = datetime.utcnow()

# Update countdown until refresh (will stop at 0 because Streamlit will refresh)
while True:
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    remaining = max(refresh_interval_sec - elapsed, 0)
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    countdown_placeholder.markdown(f"**Time until next refresh: {minutes:02d}:{seconds:02d}**")
    if remaining <= 0:
        break
    time.sleep(1)

# ------------------- Quick Access (Collapsible Footer) -------------------
st.markdown("---")

with st.expander("Quick Access Links (click to expand)"):
    st.markdown(
        """
**[TradingView](https://www.tradingview.com/)**  
**[Wealthsimple](https://my.wealthsimple.com/app/home)**  
**[Option Screener](https://optionmoves.com/screener?ticker=SPY%2C+NVDA%2C+AAPL%2C+MSFT%2C+GOOG%2C+AMZN%2C+META%2C+BRK.B%2C+TSLA%2C+AVGO%2C+LLY%2C+JPM%2C+UNH%2C+V%2C+MA%2C+JNJ%2C+XOM%2C+CVX%2C+PG%2C+PEP%2C+KO%2C+WMT%2C+BAC%2C+PFE%2C+NFLX%2C+ORCL%2C+ADBE%2C+INTC%2C+COST%2C+ABT%2C+VZ&strategy=put-credit-spread&expiryType=dte&dte=30&deltaStrikeType=delta&delta=0.30&spreadWidth=5)**  
**[IV Rank Check](https://marketchameleon.com/)**  
[Trading Plan PDF](./Small Account Put Credit Spread Plan 3.0.pdf)
"""
    )
