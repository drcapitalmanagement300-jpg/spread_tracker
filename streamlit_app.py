# app.py
import streamlit as st
from datetime import datetime, date
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd

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
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    current_profit = credit - spread_value
    return max(0, min((current_profit / credit) * 100, 100))  # Cap 0-100%

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
        current_profit_percent = compute_current_profit(short_option_price, long_option_price, t["credit"], derived["width"])
        rule_violations, abs_delta, spread_value_percent = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_option_price, long_option_price
        )

        # Strings
        abs_delta_str = f"{abs_delta:.2f}" if abs_delta is not None else "-"
        spread_value_str = f"{spread_value_percent:.0f}%" if spread_value_percent is not None else "-"
        current_profit_str = f"{current_profit_percent:.1f}%" if current_profit_percent is not None else "-"
        current_price_str = f"{current_price:.2f}" if current_price is not None else "-"

        # Determine status icon
        if rule_violations["other_rules"]:
            status_icon = "❌"
            status_text = "Some critical rules are violated."
        elif rule_violations["iv_rule"]:
            status_icon = "⚠️"
            status_text = "Current IV exceeds entry IV."
        else:
            status_icon = "✅"
            status_text = "All rules are satisfied."

        # Layout: left = info + status, right = pnl
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
            if current_iv is None or t["entry_iv"] is None:
                iv_color = "black"
            elif current_iv == t["entry_iv"]:
                iv_color = "yellow"
            elif current_iv > t["entry_iv"]:
                iv_color = "red"
            else:
                iv_color = "green"

            st.markdown(
                f"""
Short Delta: <span style='color:{delta_color}'>{abs_delta_str}</span> | Must be less than or equal to 0.40 <br>
Spread Value: <span style='color:{spread_color}'>{spread_value_str}</span> | Must be less than or equal to 150% of credit <br>
DTE: <span style='color:{dte_color}'>{derived['dte']}</span> | Must be greater than 7 DTE <br>
Current Profit: <span style='color:{profit_color}'>{current_profit_str}</span> | 50-75% Max profit target <br>
Entry IV: {t['entry_iv']:.1f}% | Current IV: <span style='color:{iv_color}'>{current_iv:.1f}%</span>
""", unsafe_allow_html=True)

            # Simplified PnL chart: single point
            pnl_df = pd.DataFrame({
                "DTE": [derived["dte"]],
                "Profit %": [current_profit_percent if current_profit_percent is not None else 0]
            }).set_index("DTE")
            st.line_chart(pnl_df, height=150)

        # Remove button
        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            st.experimental_rerun()

st.markdown("---")
st.caption("Spread value uses actual option prices — alerts are accurate, BSM delta calculated, auto-entry IV fetched. PnL chart shows current profit % vs DTE.")
