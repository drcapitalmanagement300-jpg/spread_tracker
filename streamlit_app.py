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

def get_short_leg_data(trade: dict):
    expiration = trade["expiration"].isoformat()
    _, puts = get_option_chain(trade["ticker"], expiration)
    if puts is None or puts.empty:
        return None, None
    short_strike = float(trade["short_strike"])
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty:
        return None, None
    delta = short_row['delta'].values[0] if 'delta' in short_row.columns else None
    iv = short_row['impliedVolatility'].values[0] * 100 if 'impliedVolatility' in short_row.columns else None
    last_price = short_row['lastPrice'].values[0] if 'lastPrice' in short_row.columns else None
    return delta, iv, last_price

def compute_spread_value(short_price, long_price, width, credit):
    """Calculate current spread P/L in % of max risk"""
    if short_price is None or long_price is None:
        return None
    spread_mark = short_price - long_price
    max_loss = width - credit
    if max_loss <= 0:
        return 0
    return (spread_mark / max_loss) * 100  # percent of max risk

def evaluate_rules(trade, derived, current_price, delta, current_iv):
    """Return status color based on all rules"""
    status = "green"  # default safe
    alerts = []

    # Rule 1: delta ≥ 0.40
    if delta is not None and delta >= 0.40:
        status = "red"
        alerts.append(f"Short delta {delta:.2f} ≥ 0.40")

    # Rule 2: price < short strike
    if current_price is not None and current_price < trade["short_strike"]:
        status = "red"
        alerts.append(f"Price {current_price:.2f} below short strike {trade['short_strike']}")

    # Rule 3: spread value ≥ 150–200% of credit
    # calculate spread value
    # For put credit: short_price - long_price = spread mark
    delta_price = current_price - trade["long_strike"] if current_price is not None else None
    spread_value_percent = None
    if delta_price is not None:
        spread_value_percent = compute_spread_value(current_price, trade["long_strike"], derived["width"], trade["credit"])
        if spread_value_percent >= 150:
            status = "red"
            alerts.append(f"Spread value {spread_value_percent:.0f}% ≥ 150% of credit")

    # Rule 4: DTE < 7
    if derived["dte"] <= 7 and status != "red":
        status = "yellow"
        alerts.append(f"DTE {derived['dte']} ≤ 7")

    # Rule 5: Current IV vs Entry IV
    entry_iv = trade.get("entry_iv")
    if entry_iv and current_iv:
        if current_iv > entry_iv and status != "red":
            status = "yellow"
            alerts.append(f"Current IV {current_iv:.1f}% > Entry IV {entry_iv:.1f}%")

    return status, alerts

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
        current_price = get_price(t['ticker'])
        delta, current_iv, short_price = get_short_leg_data(t)
        status_color, alerts = evaluate_rules(t, derived, current_price, delta, current_iv)

        # Card columns
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
            # Price
            if current_price is None:
                st.warning("Price unavailable")
            else:
                if current_price < t["short_strike"]:
                    st.error(f"Price: {current_price:.2f} 🚨 Below short strike!")
                else:
                    st.success(f"Price: {current_price:.2f}")
            # IV display
            entry_iv = t.get("entry_iv")
            if entry_iv and current_iv:
                iv_color = "green" if current_iv < entry_iv else "red"
                st.markdown(f"Entry IV: **{entry_iv:.1f}%**  Current IV: <span style='color:{iv_color}'>{current_iv:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Entry IV: **{entry_iv if entry_iv else '—'}%**  Current IV: {current_iv if current_iv else '—'}%")
            st.write(f"Short-leg delta: {delta if delta else '—'}")
            # Status badge
            if status_color == "green":
                st.success("Status: ✅ Safe")
            elif status_color == "yellow":
                st.warning("Status: ⚠ Warning")
            else:
                st.error("Status: ❌ Critical")

            # Alerts list
            if alerts:
                st.markdown("**Alerts:**")
                for a in alerts:
                    st.write(f"- {a}")

        with cols[3]:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.trades.pop(i)
                st.experimental_rerun()

        # Notes / Details
        with st.expander("Details / Notes"):
            st.write("Notes:", t.get("notes") or "-")
            st.write("Created:", t.get("created_at"))

st.markdown("---")
st.caption("Step 3 complete — Rules engine added with color-coded alerts. Next step: SMS notifications.")
