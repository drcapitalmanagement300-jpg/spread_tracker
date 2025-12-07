# app.py (live updating Put Credit Spread Monitor)
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
from streamlit_autorefresh import st_autorefresh

# Persistence (Google OAuth + Drive) — provided in persistence.py
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

# ----------------------------- Streamlit setup -----------------------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ----------------------------- Auto-refresh -----------------------------
REFRESH_INTERVAL = 1_000  # milliseconds
st_autorefresh(interval=REFRESH_INTERVAL, key="live_refresh")

# ----------------------------- OAuth & Drive setup -----------------------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth could not be completed. Running locally.")

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
            st.success("Logged out (local). Reload the page to sign in again.")
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ----------------------------- Helpers -----------------------------
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    elif option_type.lower() == 'put':
        return norm.cdf(d1) - 1
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
    return delta, short_price, iv

def get_long_leg_data(trade: dict):
    long_price, _ = get_leg_data(trade["ticker"], trade["expiration"], float(trade["long_strike"]), 'put')
    return long_price

def compute_current_profit(short_price, long_price, credit, width):
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    current_profit = credit - spread_value
    return max(0, min((current_profit / credit) * 100, 100))

def evaluate_rules(trade, derived, current_price, delta, short_option_price, long_option_price):
    rule_violations = {"other_rules": False, "iv_rule": False}
    abs_delta = abs(delta) if delta is not None else None
    if abs_delta is not None and abs_delta >= 0.40:
        rule_violations["other_rules"] = True
    spread_value_percent = None
    if short_option_price is not None and long_option_price is not None:
        spread_mark = short_option_price - long_option_price
        max_loss = derived["width"] - trade["credit"]
        spread_value_percent = (spread_mark / max_loss) * 100 if max_loss > 0 else None
        if spread_value_percent is not None and spread_value_percent >= 150:
            rule_violations["other_rules"] = True
    if derived["dte"] <= 7:
        rule_violations["other_rules"] = True
    return rule_violations, abs_delta, spread_value_percent

def fetch_short_iv(ticker, short_strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    return short_row['impliedVolatility'].values[0] * 100

# ----------------------------- Initialize session -----------------------------
init_state()

# ----------------------------- Add trade form -----------------------------
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
                "last_update": datetime.utcnow(),
                "cached": {}
            }
            st.session_state.trades.append(trade)
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
st.subheader("Active Trades")

# ----------------------------- Active trades -----------------------------
UPDATE_INTERVAL = 10*60  # seconds
now = datetime.utcnow()

if not st.session_state.trades:
    st.info("No trades added yet. Use the form above to add your first spread.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)

        # Check if we need to refresh data
        elapsed = (now - t.get("last_update", now)).total_seconds()
        if elapsed >= UPDATE_INTERVAL or not t.get("cached"):
            current_price = get_price(t['ticker'])
            delta, short_option_price, _ = get_short_leg_data(t)
            long_option_price = get_long_leg_data(t)
            current_profit_percent = compute_current_profit(short_option_price, long_option_price, t["credit"], derived["width"])
            rule_violations, abs_delta, spread_value_percent = evaluate_rules(
                t, derived, current_price, delta, short_option_price, long_option_price
            )
            # Cache results
            t["cached"] = {
                "current_price": current_price,
                "delta": delta,
                "short_option_price": short_option_price,
                "long_option_price": long_option_price,
                "current_profit_percent": current_profit_percent,
                "rule_violations": rule_violations,
                "spread_value_percent": spread_value_percent
            }
            t["last_update"] = now
        else:
            cached = t["cached"]
            current_price = cached["current_price"]
            delta = cached["delta"]
            short_option_price = cached["short_option_price"]
            long_option_price = cached["long_option_price"]
            current_profit_percent = cached["current_profit_percent"]
            rule_violations = cached["rule_violations"]
            spread_value_percent = cached["spread_value_percent"]

        # Countdown
        seconds_to_next_update = max(0, UPDATE_INTERVAL - (now - t["last_update"]).total_seconds())
        minutes = int(seconds_to_next_update // 60)
        seconds = int(seconds_to_next_update % 60)

        trade_container = st.container()
        countdown_placeholder = trade_container.empty()
        card_placeholder = trade_container.empty()
        chart_placeholder = trade_container.empty()

        countdown_placeholder.markdown(f"**Next update in:** {minutes}:{seconds:02d}")

        abs_delta_str = f"{abs(delta):.2f}" if delta is not None else "-"
        spread_value_str = f"{spread_value_percent:.0f}%" if spread_value_percent is not None else ""
        current_profit_str = f"{current_profit_percent:.1f}%" if current_profit_percent is not None else ""
        current_price_str = f"{current_price:.2f}" if current_price is not None else "-"

        status_icon = "❌" if rule_violations["other_rules"] else "✅"
        status_text = "Some critical rules are violated." if rule_violations["other_rules"] else "All rules are satisfied."

        card_placeholder.markdown(
            f"""
<div style='background-color:rgba(0,100,0,0.1); padding:15px; border-radius:10px;'>
Ticker: {t['ticker']}  <br>
Underlying Price: {current_price_str}  <br>
Short Strike: {t['short_strike']}  <br>
Long Strike: {t['long_strike']}  <br>
Spread Width: {derived['width']}  <br>
Expiration Date: {t['expiration']}  <br>
Current DTE: {derived['dte']}  <br>
Max Gain: {format_money(derived['max_gain'])}  <br>
Max Loss: {format_money(derived['max_loss'])}  <br>
Short Delta: {abs_delta_str} | Must be <= 0.40 <br>
Spread Value: {spread_value_str} | Must be <= 150% of credit <br>
DTE: {derived['dte']} | Must be > 7 <br>
Current Profit: {current_profit_str} | 50-75% max profit target <br>
Status: {status_icon} {status_text}
</div>
""", unsafe_allow_html=True)

        # PnL chart
        dte_range = list(range(derived["dte"] + 1))
        profit_values = [current_profit_percent if current_profit_percent is not None else 0]*len(dte_range)
        pnl_df = pd.DataFrame({"DTE": dte_range, "Profit %": profit_values})

        base_chart = alt.Chart(pnl_df).mark_line(point=True).encode(
            x=alt.X('DTE', title='Days to Expiration', scale=alt.Scale(domain=(derived["dte"], 0))),
            y=alt.Y('Profit %', title='Current Profit %', scale=alt.Scale(domain=(0,100), nice=False),
                    axis=alt.Axis(tickMinStep=10, tickCount=11))
        ).properties(height=250)

        line_50 = alt.Chart(pd.DataFrame({'y':[50]})).mark_rule(stroke="green", strokeDash=[5,5]).encode(y='y')
        line_75 = alt.Chart(pd.DataFrame({'y':[75]})).mark_rule(stroke="green", strokeDash=[5,5]).encode(y='y')
        vline = alt.Chart(pd.DataFrame({'DTE':[derived['dte']]})).mark_rule(stroke="green", strokeDash=[5,5]).encode(x='DTE')

        final_chart = base_chart + line_50 + line_75 + vline
        chart_placeholder.altair_chart(final_chart, use_container_width=True)

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            if drive_service:
                try:
                    save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    pass
            try:
                st.experimental_rerun()
            except Exception:
                pass
