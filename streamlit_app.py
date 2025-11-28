# app.py
import streamlit as st
from datetime import datetime, date
from math import isfinite
import yfinance as yf

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("Options Spread Monitor")

# ------------------- Helpers ---------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = []  # each trade is a dict

def days_to_expiry(expiry_date: date) -> int:
    return max((expiry_date - date.today()).days, 0)

def compute_derived(trade: dict) -> dict:
    short = float(trade["short_strike"])
    long = float(trade["long_strike"])
    credit = float(trade.get("credit", 0) or 0)
    width = abs(long - short)
    max_gain = credit
    max_loss = max(width - credit, 0)
    breakeven = short + credit if trade.get("type", "put_credit") == "put_credit" else short - credit
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
    """Fetch latest underlying price"""
    try:
        data = yf.Ticker(ticker).fast_info
        return float(data["last_price"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiration: str):
    """Return option chain for given expiry as dataframe"""
    try:
        ticker_obj = yf.Ticker(ticker)
        opt_chain = ticker_obj.option_chain(expiration)
        return opt_chain.calls, opt_chain.puts
    except Exception:
        return None, None

def get_short_leg_data(trade: dict):
    """Return delta and IV for short strike"""
    expiration = trade["expiration"].isoformat()
    _, puts = get_option_chain(trade["ticker"], expiration)
    if puts is None or puts.empty:
        return None, None
    # match exact strike
    short_strike = float(trade["short_strike"])
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty:
        return None, None
    delta = short_row['delta'].values[0] if 'delta' in short_row.columns else None
    iv = short_row['impliedVolatility'].values[0] * 100 if 'impliedVolatility' in short_row.columns else None
    return delta, iv

# ------------------- Initialize -------------------
init_state()

# ------------------- Left: Add trade form -------------------
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
        entry_iv = st.number_input("Entry volatility (IV%) — optional", min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
    with col3:
        entry_date = st.date_input("Entry date", value=date.today())
        notes = st.text_input("Notes (optional)")
        st.write("")  # spacing

    submitted = st.form_submit_button("Add trade for monitoring")
    if submitted:
        if not ticker:
            st.warning("Please provide a ticker symbol.")
        elif long_strike >= short_strike:
            st.warning("For a put credit spread, long strike should be LOWER than short strike. (long < short)")
        else:
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv": float(entry_iv) if entry_iv > 0 else None,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)
            st.success(f"Added {ticker} {short_strike}/{long_strike} exp {expiration}")

st.markdown("---")

# ------------------- Right: Active trades dashboard -------------------
st.subheader("Active Trades")
if not st.session_state.trades:
    st.info("No trades added yet. Use the form on the left to add your first spread.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
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
            st.write("**Live Data**")
            current_price = get_price(t['ticker'])
            if current_price is None:
                st.warning("Price unavailable")
            else:
                if current_price > t["short_strike"]:
                    st.success(f"Price: {current_price:.2f}")
                else:
                    st.error(f"Price: {current_price:.2f} 🚨 Below short strike!")

            # Fetch short-leg delta and IV
            delta, current_iv = get_short_leg_data(t)
            entry_iv = t.get("entry_iv")
            # IV display with color logic
            if entry_iv and current_iv:
                iv_color = "green" if current_iv < entry_iv else "red"
                st.markdown(f"Entry IV: **{entry_iv:.1f}%**  Current IV: <span style='color:{iv_color}'>{current_iv:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Entry IV: **{entry_iv if entry_iv else '—'}%**  Current IV: {current_iv if current_iv else '—'}%")
            st.write(f"Short-leg delta: {delta if delta else '—'}")
        with cols[3]:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.trades.pop(i)
                st.experimental_rerun()
        # Notes / Alerts
        with st.expander("Details / Alerts"):
            st.write("Notes:", t.get("notes") or "-")
            st.write("Created:", t.get("created_at"))

st.markdown("---")
st.caption("Step 2 complete — Live price, IV, and delta added. Next step: implement rules engine for alerts and take-profit / stop-loss logic.")
