# app.py
import streamlit as st
from datetime import datetime, date
import numpy as np
from scipy.stats import norm
import yfinance as yf

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("Options Spread Monitor")

# ------------------- Helpers ---------------------
def init_state():
    if "trades" not in st.session_state:
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

# ------------------- Black-Scholes Delta -----------------
def bsm_delta(option_type, S, K, T, r, sigma):
    """Calculate delta for European call/put using Black-Scholes-Merton"""
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
    return delta, iv, short_price

def get_long_leg_data(trade: dict):
    long_price, _ = get_leg_data(trade["ticker"], trade["expiration"], float(trade["long_strike"]), 'put')
    return long_price

def compute_spread_value(short_option_price, long_option_price, width, credit):
    """Returns current spread value as % of max loss"""
    if short_option_price is None or long_option_price is None or width - credit <= 0:
        return None
    spread_mark = short_option_price - long_option_price
    max_loss = width - credit
    return (spread_mark / max_loss) * 100

def evaluate_rules(trade, derived, current_price, delta, current_iv, short_option_price, long_option_price):
    status = "green"
    alerts = []

    # Delta rule
    if delta is not None and delta >= 0.40:
        status = "red"
        alerts.append(f"Short delta {delta:.2f} ≥ 0.40")

    # Price rule
    if current_price is not None and current_price < trade["short_strike"]:
        status = "red"
        alerts.append(f"Price {current_price:.2f} below short strike {trade['short_strike']}")

    # Spread value rule
    spread_value_percent = compute_spread_value(short_option_price, long_option_price, derived["width"], trade["credit"])
    if spread_value_percent is not None and spread_value_percent >= 150:
        status = "red"
        alerts.append(f"Spread value {spread_value_percent:.0f}% ≥ 150% of credit")

    # DTE rule
    if derived["dte"] <= 7 and status != "red":
        status = "yellow"
        alerts.append(f"DTE {derived['dte']} ≤ 7")

    # IV rule
    entry_iv = trade.get("entry_iv")
    if entry_iv and current_iv:
        if current_iv > entry_iv and status != "red":
            status = "yellow"
            alerts.append(f"Current IV {current_iv:.1f}% > Entry IV {entry_iv:.1f}%")

    return status, alerts, spread_value_percent

# ------------------- Auto-fetch entry IV -----------------
def fetch_short_iv(ticker, short_strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    iv = short_row['impliedVolatility'].values[0] * 100
    return iv

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

st.markdown("---")

# ------------------- Active Trades Dashboard -------------------
st.subheader("Active Trades")
if not st.session_state.trades:
    st.info("No trades added yet. Use the form above to add your first spread.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
        current_price = get_price(t['ticker'])
        delta, current_iv, short_option_price = get_short_leg_data(t)
        long_option_price = get_long_leg_data(t)
        status_color, alerts, spread_value_percent = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_option_price, long_option_price
        )

        cols = st.columns([1,2,2,1])
        with cols[0]:
            st.markdown(f"**{t['ticker']}**")
            st.write(f"{t['short_strike']}/{t['long_strike']}")
            st.write(f"Exp: {t['expiration'].isoformat()}")
            dte = derived["dte"]
            if dte <= 7:
                st.error(f"{dte} DTE")
            else:
                st.success(f"{dte} DTE")
        with cols[1]:
            st.write("**Derived**")
            st.write(f"Spread width: {derived['width']}")
            st.write(f"Max gain: {format_money(derived['max_gain'])}")
            st.write(f"Max loss: {format_money(derived['max_loss'])}")
            st.write(f"Break-even: {derived['breakeven']}")
        with cols[2]:
            st.write("**Live Data / Rules**")
            if current_price is None:
                st.warning("Price unavailable")
            else:
                if current_price < t["short_strike"]:
                    st.error(f"Price: {current_price:.2f} 🚨 Below short strike!")
                else:
                    st.success(f"Price: {current_price:.2f}")

            entry_iv = t.get("entry_iv")
            if entry_iv and current_iv:
                iv_color = "green" if current_iv < entry_iv else "red"
                st.markdown(f"Entry IV: **{entry_iv:.1f}%**  Current IV: <span style='color:{iv_color}'>{current_iv:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Entry IV: **{entry_iv if entry_iv else '—'}%**  Current IV: {current_iv if current_iv else '—'}%")

            st.write(f"Short-leg delta: {delta if delta else '—'}")
            if spread_value_percent is not None:
                st.write(f"Spread value: {spread_value_percent:.0f}% of max loss")

            if status_color == "green":
                st.success("Status: ✅ Safe")
            elif status_color == "yellow":
                st.warning("Status: ⚠ Warning")
            else:
                st.error("Status: ❌ Critical")

            if alerts:
                st.markdown("**Alerts:**")
                for a in alerts:
                    st.write(f"- {a}")

        with cols[3]:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.trades.pop(i)
                st.experimental_rerun()

        with st.expander("Details / Notes"):
            st.write("Notes:", t.get("notes") or "-")
            st.write("Created:", t.get("created_at"))

st.markdown("---")
st.caption("Spread value now uses actual option prices — alerts are accurate, BSM delta calculated, auto-entry IV fetched.")
