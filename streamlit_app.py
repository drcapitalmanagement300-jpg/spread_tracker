import streamlit as st
from datetime import date, datetime
import time
from typing import Optional
import pandas as pd
import altair as alt

from streamlit_autorefresh import st_autorefresh

# ---------------- Persistence (Drive JSON) ----------------
# Assuming persistence.py is in the same directory
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
# Interval is in milliseconds: 10 minutes = 600,000 ms
st_autorefresh(interval=600_000, key="ui_refresh")

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
header_col1, header_col2, header_col3 = st.columns([2, 6, 1])

with header_col1:
    # Displaying the uploaded logo
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.jpg", width=160)
    except Exception:
        st.write("**DR CAPITAL**")

with header_col2:
    st.title("Put Credit Spread Monitor")
    st.caption("Strategic Options Management System")

with header_col3:
    st.write("") # Spacer
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

def update_pnl_history(trade: dict, profit: Optional[float], dte: int) -> bool:
    """
    Appends today's data to history if not present. 
    Returns True if history was modified.
    """
    today = date.today().isoformat()
    trade.setdefault("pnl_history", [])
    
    # Check if today is already recorded
    if any(p.get("date") == today for p in trade["pnl_history"]):
        return False
        
    # Append new point
    trade["pnl_history"].append({
        "date": today,
        "dte": dte,
        "profit": profit if profit is not None else 0.0 # Store 0.0 if None for graph continuity
    })
    return True

# ---------------- Load Drive State ----------------
def init_state():
    if "trades" not in st.session_state:
        trades = []
        if drive_service:
            trades = load_from_drive(drive_service) or []

        for t in trades:
            t.setdefault("cached", {})
            t.setdefault("pnl_history", [])

        st.session_state.trades = trades

init_state()

# ---------------- Add Trade (No Notes) ----------------
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
                # Notes field removed as requested
                "created_at": datetime.utcnow().isoformat(),
                "cached": {},
                "pnl_history": []
            }
            st.session_state.trades.append(trade)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success(f"Position initialized for {ticker}. Market data syncing...")

st.markdown("---")

# ---------------- Display Trades (CARD UI) ----------------
st.subheader("Active Portfolio")

if not st.session_state.trades:
    st.info("No active trades.")
else:
    history_changed = False

    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})

        dte = days_to_expiry(t["expiration"])
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"]
        max_loss = width - t["credit"]

        current_price = cached.get("current_price")
        abs_delta = cached.get("abs_delta")
        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        # Update History Logic
        if update_pnl_history(t, profit_pct, dte):
            history_changed = True

        status_ok = not rules.get("other_rules", False)
        status_icon = "✅" if status_ok else "⚠️"
        card_border = "2px solid #e0e0e0" if status_ok else "2px solid #ff4b4b"

        cols = st.columns([3, 4])

        # -------- LEFT CARD (Details) --------
        with cols[0]:
            st.markdown(
                f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: {card_border}; height: 100%;'>
                    <h3 style='margin-top:0; color: #333;'>{t['ticker']} <span style='font-size: 16px; color: #666;'>{status_icon}</span></h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 14px;'>
                        <div><strong>Short:</strong> {t['short_strike']}</div>
                        <div><strong>Long:</strong> {t['long_strike']}</div>
                        <div><strong>Exp:</strong> {t['expiration']}</div>
                        <div><strong>DTE:</strong> {dte}</div>
                        <div><strong>Max Gain:</strong> {format_money(max_gain)}</div>
                        <div><strong>Max Loss:</strong> {format_money(max_loss)}</div>
                        <div style='grid-column: span 2; margin-top: 8px;'><strong>Underlying:</strong> {format_money(current_price) if current_price else '-'}</div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

        # -------- RIGHT CARD (Metrics & Chart) --------
        with cols[1]:
            # Color logic
            delta_color = "#d32f2f" if abs_delta and abs_delta >= 0.40 else "#2e7d32"
            spread_color = "#d32f2f" if spread_value and spread_value >= 150 else "#2e7d32"
            dte_color = "#d32f2f" if dte <= 7 else "#2e7d32"

            if profit_pct is None:
                profit_color = "#333"
            elif profit_pct < 50:
                profit_color = "#2e7d32"
            elif profit_pct <= 75:
                profit_color = "#f9a825"
            else:
                profit_color = "#d32f2f"

            # Metrics Row
            st.markdown(
                f"""
                <div style='display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 15px;'>
                    <span>Short Delta: <strong style='color:{delta_color}'>{abs_delta if abs_delta else "-"}</strong></span>
                    <span>Spread Val: <strong style='color:{spread_color}'>{spread_value if spread_value else "-"}%</strong></span>
                    <span>Profit: <strong style='color:{profit_color}'>{profit_pct if profit_pct is not None else "-"}%</strong></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------- PnL CHART --------
            if t["pnl_history"]:
                df = pd.DataFrame(t["pnl_history"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                # Base chart
                base = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")),
                    y=alt.Y("profit:Q", scale=alt.Scale(domain=[-100, 100]), title="Profit %"),
                    tooltip=["date", "profit"]
                ).properties(height=200)

                # Horizontal Rules (Green as requested)
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

    if history_changed and drive_service:
        save_to_drive(drive_service, st.session_state.trades)

st.markdown("---")
st.caption("Market data is populated externally and stored in Drive JSON.")

# ---------------- Manual Controls (Restored) ----------------
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
        loaded = []
        if drive_service:
            try:
                loaded = load_from_drive(drive_service) or []
            except Exception:
                loaded = []
        if loaded:
            for tr in loaded:
                tr.setdefault("pnl_history", [])
            st.session_state.trades = loaded
            st.success("Loaded trades from Drive.")
            st.experimental_rerun()
        else:
            st.info("No trades found or load failed.")

# ---------------- Countdown Timer (Restored) ----------------
# NOTE: This loop runs at the end of the script execution.
# It provides the visual countdown requested.

refresh_interval_sec = 600  # 10 minutes match st_autorefresh
countdown_placeholder = st.empty()
start_time = datetime.utcnow()

while True:
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    remaining = max(refresh_interval_sec - elapsed, 0)
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    
    countdown_placeholder.markdown(
        f"""
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            Next auto-refresh in: <strong>{minutes:02d}:{seconds:02d}</strong>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    if remaining <= 0:
        break
    time.sleep(1)
