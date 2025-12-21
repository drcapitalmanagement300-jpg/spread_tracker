# app.py  (SCRIPT 3 — Drive-backed logic + Card UI)

import streamlit as st
from datetime import date, datetime
from typing import Optional
import pandas as pd
import altair as alt

from streamlit_autorefresh import st_autorefresh

# ---------------- Persistence (Drive JSON) ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

st_autorefresh(interval=600_000, key="ui_refresh")  # UI refresh only

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

_, logout_col = st.columns([9, 1])
with logout_col:
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
        st.experimental_rerun()

# ---------------- Helpers ----------------
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

# ---------------- Add Trade (no market math) ----------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Ticker").upper()
        short_strike = st.number_input("Short strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long strike", min_value=0.0, format="%.2f")
    with c2:
        expiration = st.date_input("Expiration date")
        credit = st.number_input("Credit received", min_value=0.0, format="%.2f")
    with c3:
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

# ---------------- Display Trades (CARD UI) ----------------
st.subheader("Active Trades")

if not st.session_state.trades:
    st.info("No trades added.")
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

        if update_pnl_history(t, profit_pct, dte):
            history_changed = True

        status_ok = not rules.get("other_rules", False)
        status_icon = "✅" if status_ok else "❌"

        cols = st.columns([3, 3])

        # -------- LEFT CARD --------
        with cols[0]:
            st.markdown(
                f"""
<div style='background-color:rgba(0,100,0,0.08);
            padding:16px; border-radius:12px'>
<b>{t['ticker']}</b><br>
Underlying: {current_price or "-"}<br>
Short: {t['short_strike']}<br>
Long: {t['long_strike']}<br>
Width: {width}<br>
Expiration: {t['expiration']}<br>
DTE: {dte}<br>
Max Gain: {format_money(max_gain)}<br>
Max Loss: {format_money(max_loss)}
</div>
""", unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:20px;margin-top:8px'>{status_icon}</div>", unsafe_allow_html=True)

        # -------- RIGHT CARD --------
        with cols[1]:
            delta_color = "red" if abs_delta and abs_delta >= 0.40 else "green"
            spread_color = "red" if spread_value and spread_value >= 150 else "green"
            dte_color = "red" if dte <= 7 else "green"

            if profit_pct is None:
                profit_color = "black"
            elif profit_pct < 50:
                profit_color = "green"
            elif profit_pct <= 75:
                profit_color = "orange"
            else:
                profit_color = "red"

            st.markdown(
                f"""
Short Delta: <span style='color:{delta_color}'>{abs_delta or "-"}</span><br>
Spread Value: <span style='color:{spread_color}'>{spread_value or "-"}</span>%<br>
DTE: <span style='color:{dte_color}'>{dte}</span><br>
Current Profit: <span style='color:{profit_color}'>{profit_pct or "-"}</span>%
""",
                unsafe_allow_html=True
            )

            # -------- PnL CHART --------
            if t["pnl_history"]:
                df = pd.DataFrame(t["pnl_history"])
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")

                base = alt.Chart(df).mark_line(point=True).encode(
                    x="date:T",
                    y=alt.Y("profit:Q", scale=alt.Scale(domain=[0, 100]), title="Profit %")
                )

                line_50 = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(strokeDash=[5,5]).encode(y="y")
                line_75 = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(strokeDash=[5,5]).encode(y="y")

                st.altair_chart(base + line_50 + line_75, use_container_width=True)

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.experimental_rerun()

    if history_changed and drive_service:
        save_to_drive(drive_service, st.session_state.trades)

st.markdown("---")
st.caption("Market data is populated externally and stored in Drive JSON.")
