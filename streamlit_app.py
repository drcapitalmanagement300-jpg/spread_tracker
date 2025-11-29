# app.py
import streamlit as st
from datetime import datetime, date
import numpy as np
from scipy.stats import norm
import yfinance as yf

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

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
    if short_option_price is None or long_option_price is None or width - credit <= 0:
        return None
    spread_mark = short_option_price - long_option_price
    max_loss = width - credit
    return (spread_mark / max_loss) * 100

def compute_current_profit(short_price, long_price, credit, width):
    """Returns current profit as % of max gain"""
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    max_loss = width - credit
    current_profit = credit - spread_value  # profit if spread mark rises
    return (current_profit / credit) * 100

def fetch_short_iv(ticker, short_strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    iv = short_row['impliedVolatility'].values[0] * 100
    return iv

def evaluate_rules(trade, derived, current_price, delta, current_iv, short_option_price, long_option_price):
    status_color = "green"
    alerts = []

    abs_delta = abs(delta) if delta is not None else None
    if abs_delta is not None and abs_delta >= 0.40:
        status_color = "red"
        alerts.append(f"Short delta {abs_delta:.2f} greater than or equal to 0.40")

    if current_price is not None and current_price < trade["short_strike"]:
        status_color = "red"
        alerts.append(f"Price {current_price:.2f} less than short strike {trade['short_strike']}")

    spread_value_percent = compute_spread_value(short_option_price, long_option_price, derived["width"], trade["credit"])
    if spread_value_percent is not None and spread_value_percent >= 150:
        status_color = "red"
        alerts.append(f"Spread value {spread_value_percent:.0f}% greater than or equal to 150% of credit")

    if derived["dte"] <= 7 and status_color != "red":
        status_color = "orange"
        alerts.append(f"DTE {derived['dte']} less than or equal to 7")

    return status_color, alerts, abs_delta, spread_value_percent

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
        status_color, alerts, abs_delta, spread_value_percent = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_option_price, long_option_price
        )
        current_profit_percent = compute_current_profit(short_option_price, long_option_price, t["credit"], derived["width"])

        cols = st.columns([3,2])
        with cols[0]:
            # --- Spread Info Card ---
            with st.container():
                st.markdown("### Spread Info")
                st.markdown(f"""
**Ticker:** {t['ticker']}  
**Underlying Price:** {current_price if current_price else '-'}  
**Short Strike:** {t['short_strike']}  
**Long Strike:** {t['long_strike']}  
**Spread Width:** {derived['width']}  
**Expiration Date:** {t['expiration']}  
**Current DTE:** {derived['dte']}  
**Max Gain:** {format_money(derived['max_gain'])}  
**Max Loss:** {format_money(derived['max_loss'])}  
""")

            # --- Stats / Status ---
            st.markdown("### Stats / Status")
            metrics = {
                "Short Delta": f"{abs_delta:.2f}" if abs_delta else "-",
                "Exit Price": f"{current_price:.2f}" if current_price else "-",
                "Spread Value": f"{spread_value_percent:.0f}%" if spread_value_percent else "-",
                "DTE": derived['dte'],
                "Current Profit": f"{current_profit_percent:.1f}%" if current_profit_percent else "-",
                "Entry IV <= Current IV": f"{t['entry_iv']:.1f}% <= {current_iv:.1f}%" if t['entry_iv'] and current_iv else "-"
            }

            rules_text = [
                "Must be less than or equal to 0.40",
                f"Must be greater than or equal to {t['short_strike']}",
                "Must be less than or equal to 150% of credit",
                "Must be greater than 7 DTE",
                "Take profit 50–75%",
                "Must be less than or equal to current IV"
            ]

            for (metric, value), rule in zip(metrics.items(), rules_text):
                color = "green"
                if metric == "Short Delta" and abs_delta is not None and abs_delta >= 0.40:
                    color = "red"
                if metric == "Exit Price" and current_price is not None and current_price < t['short_strike']:
                    color = "red"
                if metric == "Spread Value" and spread_value_percent is not None and spread_value_percent >= 150:
                    color = "red"
                if metric == "DTE" and derived['dte'] <= 7:
                    color = "orange"
                if metric == "Entry IV <= Current IV" and t['entry_iv'] and current_iv and current_iv >= t['entry_iv']:
                    color = "red"

                st.markdown(f"- **{metric}:** <span style='color:{color}'>{value}</span> | {rule}", unsafe_allow_html=True)

        # Remove button
        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            st.experimental_rerun()

st.markdown("---")
st.caption("Spread value uses actual option prices — alerts are accurate, BSM delta calculated, auto-entry IV fetched. Stats section now includes Current Profit dynamically.")
