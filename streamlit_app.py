import streamlit as st
from datetime import date, datetime
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
        <h1 style='margin-bottom: 0px;'>Put Credit Spread Monitor</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
        st.experimental_rerun()

st.markdown("---")

# ---------------- Helpers ----------------
def days_to_expiry(expiry) -> int:
    try:
        expiry = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
        return max((expiry - date.today()).days, 0)
    except Exception:
        return 0

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

def get_entry_dte(entry_date_str, expiry_date_str):
    try:
        return (date.fromisoformat(expiry_date_str) - date.fromisoformat(entry_date_str)).days
    except Exception:
        return 30

def save_closed_trade_snapshot(drive_service, trade, notes):
    snapshot = {
        "closed_at": datetime.utcnow().isoformat(),
        "notes": notes,
        "trade_snapshot": trade,
    }
    if drive_service:
        filename = f"closed_trade_{trade['id']}_{datetime.utcnow().isoformat()}.json"
        save_to_drive(drive_service, snapshot, filename_override=filename)

# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    st.session_state.setdefault("trades", [])

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
                "pnl_history": [],
            }
            st.session_state.trades.append(trade)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success(f"Position initialized for {ticker}.")

st.markdown("---")

# ---------------- Display Trades ----------------
st.subheader("Active Portfolio")

for i, t in enumerate(st.session_state.trades):
    cached = t.get("cached", {})

    current_dte = days_to_expiry(t["expiration"])
    entry_dte = get_entry_dte(t["entry_date"], t["expiration"])
    width = abs(t["short_strike"] - t["long_strike"])
    max_gain = t["credit"]
    max_loss = width - t["credit"]

    abs_delta = cached.get("abs_delta")
    spread_value = cached.get("spread_value_percent")
    profit_pct = cached.get("current_profit_percent")
    day_change = cached.get("day_change_pct")

    # Intraday change formatting
    if day_change is None:
        day_change_html = ""
    else:
        arrow = "▲" if day_change >= 0 else "▼"
        color = "green" if day_change >= 0 else "#d32f2f"
        day_change_html = f"<span style='color:{color}; font-size:16px;'> {day_change:.1f}% {arrow}</span>"

    cols = st.columns([3, 4])

    # -------- LEFT CARD --------
    with cols[0]:
        st.markdown(f"""
        <div style="line-height:1.4; font-size:15px;">
            <h3 style="margin-bottom:5px;">{t['ticker']}{day_change_html}</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr;">
                <div><strong>Short:</strong> {t['short_strike']}</div>
                <div><strong>Max Gain:</strong> {format_money(max_gain)}</div>
                <div><strong>Long:</strong> {t['long_strike']}</div>
                <div><strong>Max Loss:</strong> {format_money(max_loss)}</div>
                <div><strong>Exp:</strong> {t['expiration']}</div>
                <div><strong>Width:</strong> {width:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Close Position / Log ---
        if st.button("Close Position / Log", key=f"log_{i}"):
            st.session_state[f"closing_{i}"] = True

        if st.session_state.get(f"closing_{i}"):
            notes = st.text_area("Closing Notes / Reason", key=f"notes_{i}")
            if st.button("Submit Close Log", key=f"submit_{i}"):
                save_closed_trade_snapshot(drive_service, t, notes)
                st.session_state.trades.pop(i)
                if drive_service:
                    save_to_drive(drive_service, st.session_state.trades)
                st.experimental_rerun()

    # -------- RIGHT CARD --------
    with cols[1]:
        st.markdown(f"""
        <div style="font-size:14px;">
            <div>Short-delta: <strong>{abs_delta if abs_delta else 'Pending'}</strong>
                <span style='color:gray'>(Must not exceed 0.40)</span></div>
            <div>Spread Value: <strong>{spread_value if spread_value else 'Pending'}%</strong>
                <span style='color:gray'>(Must not exceed 150%)</span></div>
            <div>DTE: <strong>{current_dte}</strong>
                <span style='color:gray'>(Must not be less than 7)</span></div>
            <div>Profit: <strong>{profit_pct if profit_pct else 'Pending'}%</strong>
                <span style='color:gray'>(Must sell between 50–75%)</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- Manual Controls ----------------
c1, c2 = st.columns(2)

with c1:
    if st.button("Save all trades to Google Drive"):
        if drive_service:
            save_to_drive(drive_service, st.session_state.trades)
            st.success("Saved successfully.")

with c2:
    if st.button("Reload trades from Google Drive"):
        if drive_service:
            loaded = load_from_drive(drive_service)
            if loaded is not None:
                st.session_state.trades = loaded
                st.experimental_rerun()
