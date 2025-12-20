# streamline_app.py
import streamlit as st
from datetime import date, datetime
from typing import Optional
import pandas as pd
import altair as alt

from streamlit_autorefresh import st_autorefresh

# Persistence (Google OAuth + Drive)
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# UI auto-refresh (purely visual)
st_autorefresh(interval=600_000, key="ui_refresh")

# -------------------------------------------------
# Auth / Drive
# -------------------------------------------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth not fully configured. Running locally.")

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
        st.experimental_rerun()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def days_to_expiry(expiry) -> int:
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    return max((expiry - date.today()).days, 0)

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

def update_pnl_history(trade: dict, profit: Optional[float], dte: int) -> bool:
    today = date.today().isoformat()
    trade.setdefault("pnl_history", [])
    if any(p["date"] == today for p in trade["pnl_history"]):
        return False
    trade["pnl_history"].append({
        "date": today,
        "dte": dte,
        "profit": profit
    })
    return True

# -------------------------------------------------
# Load state (Drive is authoritative)
# -------------------------------------------------
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

# -------------------------------------------------
# Add Trade (creation only — no market math)
# -------------------------------------------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker").upper()
        short_strike = st.number_input("Short strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long strike", min_value=0.0, format="%.2f")
    with col2:
        expiration = st.date_input("Expiration date")
        credit = st.number_input("Credit received", min_value=0.0, format="%.2f")
    with col3:
        entry_date = st.date_input("Entry date")
        notes = st.text_input("Notes")

    submitted = st.form_submit_button("Add trade")

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
                "entry_iv": None,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat(),
                "cached": {},
                "pnl_history": []
            }
            st.session_state.trades.append(trade)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success("Trade added. Market data will populate automatically.")

st.markdown("---")

# -------------------------------------------------
# Display Trades (READ-ONLY market data)
# -------------------------------------------------
st.subheader("Active Trades")

if not st.session_state.trades:
    st.info("No trades added.")
else:
    history_changed = False

    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})

        dte = days_to_expiry(t["expiration"])
        width = abs(t["short_strike"] - t["long_strike"])

        current_price = cached.get("current_price")
        abs_delta = abs(cached.get("delta")) if cached.get("delta") is not None else None
        spread_value_percent = cached.get("spread_value_percent")
        profit_percent = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        if update_pnl_history(t, profit_percent, dte):
            history_changed = True

        status_icon = "❌" if rules.get("other_rules") else "✅"

        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"""
**{t['ticker']}**  
Underlying: {current_price or "-"}  
Short: {t['short_strike']}  
Long: {t['long_strike']}  
Width: {width}  
DTE: {dte}  
Max Gain: {format_money(t['credit'])}  
Max Loss: {format_money(width - t['credit'])}  

{status_icon}
""")

        with colB:
            st.markdown(f"""
Short Delta: {abs_delta or "-"}  
Spread Value: {spread_value_percent or "-"}%  
Current Profit: {profit_percent or "-"}%  
""")

            if t["pnl_history"]:
                df = pd.DataFrame(t["pnl_history"])
                df["date"] = pd.to_datetime(df["date"])
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(
                        x="date:T",
                        y=alt.Y("profit:Q", scale=alt.Scale(domain=[0, 100]))
                    )
                )
                st.altair_chart(chart, use_container_width=True)

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.experimental_rerun()

    if history_changed and drive_service:
        save_to_drive(drive_service, st.session_state.trades)

st.markdown("---")
st.caption("Market data updates automatically every minute via GitHub Actions.")
