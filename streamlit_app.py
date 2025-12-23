import streamlit as st
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import altair as alt

from streamlit_autorefresh import st_autorefresh

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")

# ---------------- UI Refresh ----------------
# Refresh every 1 minute (60,000 ms) to pick up changes from the GitHub Action
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth not fully configured. Running locally.")

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# ---------------- Header & Logo ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])

with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=140)
    except Exception:
        st.write("**DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Put Credit Spread Monitor</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.write("") 
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
        st.experimental_rerun()

st.markdown("---")

# ---------------- Helpers ----------------
def days_to_expiry(expiry) -> int:
    if isinstance(expiry, str):
        try:
            expiry = date.fromisoformat(expiry)
        except:
            return 0
    return max((expiry - date.today()).days, 0)

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

# ---------------- Load Drive State ----------------
# We load every run to ensure we display the latest data from the backend script
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Add Trade ----------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("New Position Entry")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Ticker").upper()
        short_strike = st.number_input("Short Strike", min_value=0.0, format="%.2f")
    with c2:
        expiration = st.date_input("Expiration Date")
        long_strike = st.number_input("Long Strike", min_value=0.0, format="%.2f")
    with c3:
        entry_date = st.date_input("Entry Date")
        credit = st.number_input("Credit Received", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Initialize Position")

    if submitted:
        if not ticker:
            st.warning("Ticker required.")
        elif long_strike >= short_strike:
            st.warning("Long strike must be lower than short strike.")
        else:
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration.isoformat(),
                "credit": credit,
                "entry_date": entry_date.isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "cached": {},
                "pnl_history": []
            }
            # Add to local state
            st.session_state.trades.append(trade)
            # Push to Drive immediately
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success(f"Position initialized for {ticker}. Backend will sync data shortly.")

st.markdown("---")

# ---------------- Display Trades ----------------
st.subheader("Active Portfolio")

if not st.session_state.trades:
    st.info("No active trades.")
else:
    for i, t in enumerate(st.session_state.trades):
        # Data is now strictly pulled from the 'cached' object populated by the backend script
        cached = t.get("cached", {})

        dte = days_to_expiry(t["expiration"])
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"]
        max_loss = width - t["credit"]

        # Backend Data
        current_price = cached.get("current_price")
        abs_delta = cached.get("abs_delta")
        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        status_ok = not rules.get("other_rules", False)
        status_icon = "✅" if status_ok else "⚠️"
        
        # Color coding
        delta_color = "red" if abs_delta and abs_delta >= 0.40 else "green"
        delta_val = f"{abs_delta:.2f}" if abs_delta is not None else "Pending"

        spread_color = "red" if spread_value and spread_value >= 150 else "green"
        spread_val = f"{spread_value:.0f}" if spread_value is not None else "Pending"

        dte_color = "red" if dte <= 7 else "green"

        if profit_pct is None:
            profit_color = "inherit"
            profit_val = "Pending"
        else:
            profit_val = f"{profit_pct:.1f}"
            if profit_pct >= 50:
                profit_color = "green"
            elif profit_pct < 0:
                profit_color = "red"
            else:
                profit_color = "#e6b800"

        cols = st.columns([3, 4])

        # -------- LEFT CARD (Details) --------
        with cols[0]:
            st.markdown(f"### {t['ticker']} {status_icon}")
            d1, d2 = st.columns(2)
            with d1:
                st.write(f"**Short:** {t['short_strike']}")
                st.write(f"**Long:** {t['long_strike']}")
                st.write(f"**Exp:** {t['expiration']}")
            with d2:
                st.write(f"**Max Gain:** {format_money(max_gain)}")
                st.write(f"**Max Loss:** {format_money(max_loss)}")
                st.write(f"**Underlying:** {format_money(current_price) if current_price else '-'}")
            st.write(f"**Width:** {width:.2f}")

        # -------- RIGHT CARD (Alerts & Chart) --------
        with cols[1]:
            st.markdown(
                f"""
                <div style="font-size: 14px; margin-bottom: 15px;">
                    <div>Short-delta is currently <strong style='color:{delta_color}'>{delta_val}</strong> | Must remain below 0.40.</div>
                    <div style="margin-top:4px;">Spread value is <strong style='color:{spread_color}'>{spread_val}%</strong> | Must remain below 150–200% of the credit received.</div>
                    <div style="margin-top:4px;">DTE is <strong style='color:{dte_color}'>{dte}</strong> | Must be sold before 7 DTE.</div>
                    <div style="margin-top:4px;">Current profit is <strong style='color:{profit_color}'>{profit_val}%</strong> | Must be sold at 50–75% of max profit.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # CHART LOGIC
            # We want the axis to span from Entry Date to Expiration Date
            try:
                entry_dt = date.fromisoformat(t["entry_date"])
                expire_dt = date.fromisoformat(t["expiration"])
            except:
                entry_dt = date.today()
                expire_dt = date.today()

            if t.get("pnl_history"):
                df = pd.DataFrame(t["pnl_history"])
                df["date"] = pd.to_datetime(df["date"])
                
                # Chart setup
                base = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X(
                        "date:T", 
                        title=None, 
                        axis=alt.Axis(format="%b %d"),
                        # FORCE the domain to be the full trade duration
                        scale=alt.Scale(domain=[pd.to_datetime(entry_dt), pd.to_datetime(expire_dt)])
                    ),
                    y=alt.Y("profit:Q", scale=alt.Scale(domain=[-100, 100]), title="Profit %"),
                    tooltip=["date", "profit"]
                ).properties(height=200)

                line_50 = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y")
                line_75 = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y")
                line_0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeWidth=1).encode(y="y")

                st.altair_chart(base + line_50 + line_75 + line_0, use_container_width=True)
            else:
                st.caption("Waiting for market data history...")

            # Remove Button
            if st.button("Close Position / Remove", key=f"remove_{i}"):
                st.session_state.trades.pop(i)
                if drive_service:
                    save_to_drive(drive_service, st.session_state.trades)
                st.experimental_rerun()

st.markdown("---")

# ---------------- Manual Controls ----------------
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
            st.error("Failed to save to Drive.")

with colB:
    if st.button("📥 Reload trades from Google Drive"):
        # Explicit reload logic
        if drive_service:
            loaded = load_from_drive(drive_service)
            if loaded is not None:
                st.session_state.trades = loaded
                st.success("Loaded trades from Drive.")
                st.experimental_rerun()
            else:
                st.info("No trades found or load failed.")

st.markdown("---")

# ---------------- External Tools Section ----------------
st.subheader("External Tools")

tools_c1, tools_c2, tools_c3, tools_c4 = st.columns(4)

with tools_c1:
    st.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)

with tools_c2:
    st.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)

with tools_c3:
    screener_url = "https://optionmoves.com/screener?ticker=SPY%2C+NVDA%2C+AAPL%2C+MSFT%2C+GOOG%2C+AMZN%2C+META%2C+BRK.B%2C+TSLA%2C+AVGO%2C+LLY%2C+JPM%2C+UNH%2C+V%2C+MA%2C+JNJ%2C+XOM%2C+CVX%2C+PG%2C+PEP%2C+KO%2C+WMT%2C+BAC%2C+PFE%2C+NFLX%2C+ORCL%2C+ADBE%2C+INTC%2C+COST%2C+ABT%2C+VZ&strategy=put-credit-spread&expiryType=dte&dte=30&deltaStrikeType=delta&delta=0.30&spreadWidth=5"
    st.link_button("Option Screener", screener_url, use_container_width=True)

with tools_c4:
    st.link_button("IV Rank Check", "https://marketchameleon.com/", use_container_width=True)
