import streamlit as st
from datetime import datetime, date
from math import isfinite
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("Options Spread Monitor")

# ------------------- Helpers ---------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = []  # each trade is a dict

def days_to_expiry(expiry):
    if isinstance(expiry, (datetime, date)):
        return (expiry - datetime.now().date()).days
    return None

# ------------------- App Start -------------------
init_state()

st.sidebar.header("Add New Trade")
ticker = st.sidebar.text_input("Ticker").upper()
short_strike = st.sidebar.number_input("Short Strike", step=0.5)
long_strike = st.sidebar.number_input("Long Strike", step=0.5)
credit = st.sidebar.number_input("Credit Received", step=0.01)
entry_price = st.sidebar.number_input("Entry Price (Underlying)", step=0.01)
entry_iv = st.sidebar.number_input("Entry IV (%)", step=0.1)
expiry = st.sidebar.date_input("Expiration Date")

if st.sidebar.button("Add Trade"):
    if ticker and short_strike and long_strike and credit:
        st.session_state.trades.append({
            "ticker": ticker,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "credit": credit,
            "entry_price": entry_price,
            "entry_iv": entry_iv,
            "expiry": expiry,
            "opened": datetime.now().date()
        })

# ------------------- Display Trades -------------------
for trade in st.session_state.trades:
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.subheader(f"{trade['ticker']} Credit Spread")
        st.write(
            f"Short Put: **{trade['short_strike']}** | "
            f"Long Put: **{trade['long_strike']}** | "
            f"Credit: **${trade['credit']}**"
        )

    with tcol2:
        if st.button(f"Remove {trade['ticker']}"):
            st.session_state.trades.remove(trade)
            st.experimental_rerun()

    # Pull market data
    data = yf.Ticker(trade["ticker"])
    price = data.history(period="1d")["Close"].iloc[-1]
    IV = data.info.get("impliedVolatility", None)
    delta = round((trade["short_strike"] - price) / price, 3)

    # Profit calculation (simple)
    distance_moved = trade["entry_price"] - price
    max_gain = trade["credit"]
    width = abs(trade["short_strike"] - trade["long_strike"])
    max_loss = width - max_gain

    current_profit = (max_gain - max(0, distance_moved)) / max_gain * 100
    current_profit = max(0, min(current_profit, 100))

    DTE = days_to_expiry(trade["expiry"])
    days_open = (datetime.now().date() - trade["opened"]).days

    # ---------------- STATS CARD ----------------
    st.markdown("---")
    stat_col1, stat_col2 = st.columns([1, 1])

    with stat_col1:
        st.write(
            f"Short Delta: **{delta}** | Must be less than or equal to 0.40\n"
            f"Spread Value: **{round((price - trade['short_strike']) / width * 100, 1)}%** "
            f"| Must be less than or equal to 150% of credit\n"
            f"DTE: **{DTE}** | Must be greater than 7 DTE\n"
            f"Current Profit: **{round(current_profit, 2)}%** | 50-75% Max profit target"
        )

    # IV coloring
    if IV is None:
        iv_color = "white"
    elif IV > trade["entry_iv"]:
        iv_color = "red"
    elif IV == trade["entry_iv"]:
        iv_color = "gold"
    else:
        iv_color = "lightgreen"

    with stat_col2:
        st.markdown(
            f"""
            Entry IV: **{trade['entry_iv']}%**  
            <span style='color:{iv_color}'>Current IV: <b>{round(IV,2) if IV else 'N/A'}%</b></span>
            """,
            unsafe_allow_html=True
        )

    # ---------------- PNL GRAPH ----------------
    st.markdown("###")

    pnl_df = pd.DataFrame({
        "Days Until Expiry": [DTE],
        "Current Profit (%)": [current_profit]
    })

    pnl_df = pnl_df.sort_values("Days Until Expiry", ascending=False)

    st.markdown("##### PnL vs Days Until Expiration")
    st.line_chart(
        pnl_df,
        x="Days Until Expiry",
        y="Current Profit (%)"
    )

    st.write("Y-axis: 0–100% | X-axis counts down to expiration (0 = expires)")

    st.markdown("---")
