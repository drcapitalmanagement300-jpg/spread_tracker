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

def get_entry_dte(entry_date_str, expiry_date_str):
    try:
        entry = date.fromisoformat(entry_date_str)
        expiry = date.fromisoformat(expiry_date_str)
        return (expiry - entry).days
    except:
        return 30 

# ---------------- Load Drive State ----------------
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
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}-{datetime.utcnow().timestamp()}",
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
            st.session_state.trades.append(trade)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success(f"Position initialized for {ticker}.")

st.markdown("---")

# ---------------- Display Trades ----------------
st.subheader("Active Portfolio")

if not st.session_state.trades:
    st.info("No active trades.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})
        current_dte = days_to_expiry(t["expiration"])
        entry_dte = get_entry_dte(t["entry_date"], t["expiration"])
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"]
        max_loss = width - t["credit"]

        # Backend Data
        current_price = cached.get("current_price")
        price_change_pct = cached.get("daily_change_percent", 0.0) # Pulled by backend script
        abs_delta = cached.get("abs_delta")
        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        # --- Ticker Display with % Change ---
        change_color = "green" if price_change_pct >= 0 else "#d32f2f"
        arrow = "↑" if price_change_pct >= 0 else "↓"
        ticker_display = f"{t['ticker']} <span style='color:{change_color}; font-size:0.8em; margin-left:10px;'>{abs(price_change_pct):.1f}% {arrow}</span>"

        # --- Status Logic ---
        status_msg = "Status Nominal"
        status_icon = "✅"
        status_color = "green"

        if rules.get("other_rules", False):
            status_icon = "⚠️"
            status_color = "#d32f2f"
            if abs_delta and abs_delta >= 0.40: status_msg = "Short Delta Exceeded (>0.40)"
            elif spread_value and spread_value >= 150: status_msg = "Spread Value High (>150%)"
            elif current_dte <= 7: status_msg = "Expiration Imminent (<7 DTE)"
        
        if profit_pct and profit_pct >= 50:
            status_icon = "💰" 
            status_msg = "Profit Target Reached"
            status_color = "green"

        # --- Color Coding ---
        delta_color = "#d32f2f" if abs_delta and abs_delta >= 0.40 else "green"
        spread_color = "#d32f2f" if spread_value and spread_value >= 150 else "green"
        dte_color = "#d32f2f" if current_dte <= 7 else "green"
        profit_color = "green" if (profit_pct or 0) >= 50 else "#e6b800"

        cols = st.columns([3, 4])

        # -------- LEFT CARD (Details) --------
        with cols[0]:
            st.markdown(f"""
            <div style="line-height: 1.2; font-size: 15px;">
                <h3 style="margin-bottom: 2px; padding-bottom:0;">{ticker_display}</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0px;">
                    <div><strong>Short:</strong> {t['short_strike']}</div>
                    <div><strong>Max Gain:</strong> {format_money(max_gain)}</div>
                    <div><strong>Long:</strong> {t['long_strike']}</div>
                    <div><strong>Max Loss:</strong> {format_money(max_loss)}</div>
                    <div><strong>Exp:</strong> {t['expiration']}</div>
                    <div><strong>Width:</strong> {width:.2f}</div>
                </div>
                <div style="margin-top: 10px; padding-top: 5px; border-top: 1px solid #eee; color: {status_color}; font-weight: bold;">
                   {status_icon} {status_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- CLOSE / LOG DIALOG ---
            if st.button("Close Position / Log", key=f"log_btn_{i}"):
                st.session_state[f"show_log_{i}"] = True

            if st.session_state.get(f"show_log_{i}"):
                with st.expander("📝 Close Trade Notes", expanded=True):
                    notes = st.text_area("Reason for closing / Trade notes:", key=f"notes_{i}")
                    if st.button("Confirm Close & Log", key=f"confirm_{i}"):
                        # Log Snapshot
                        closed_trade = t.copy()
                        closed_trade["closed_at"] = datetime.utcnow().isoformat()
                        closed_trade["closing_notes"] = notes
                        # In a real app, save to a 'closed_trades.json' here
                        
                        st.session_state.trades.pop(i)
                        if drive_service:
                            save_to_drive(drive_service, st.session_state.trades)
                        st.success("Trade logged and removed.")
                        st.rerun()

        # -------- RIGHT CARD (Alerts & Chart) --------
        with cols[1]:
            st.markdown(f"""
                <div style="font-size: 13px; line-height: 1.4;">
                    <div>Short-delta: <strong style='color:{delta_color}'>{abs_delta or 'Pending'}</strong> <span style='color:gray;'>(Must not exceed 0.40)</span></div>
                    <div>Spread Value: <strong style='color:{spread_color}'>{spread_value or 'Pending'}%</strong> <span style='color:gray;'>(Must not exceed 150%)</span></div>
                    <div>DTE: <strong style='color:{dte_color}'>{current_dte}</strong> <span style='color:gray;'>(Must not be less than 7)</span></div>
                    <div>Profit: <strong style='color:{profit_color}'>{profit_pct or 'Pending'}%</strong> <span style='color:gray;'>(Must sell between 50-75%)</span></div>
                </div>
                """, unsafe_allow_html=True)

            if t.get("pnl_history"):
                df = pd.DataFrame(t["pnl_history"])
                base = alt.Chart(df).mark_line(point=True).encode(
                    x=alt.X("dte:Q", title="DTE", scale=alt.Scale(domain=[entry_dte, 0])),
                    y=alt.Y("profit:Q", title="Profit %"),
                    tooltip=["dte", "profit"]
                ).properties(height=150)
                st.altair_chart(base, use_container_width=True)

        st.markdown("<hr style='margin: 15px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

# ---------------- Manual Controls ----------------
st.subheader("System Sync")
sync_c1, sync_c2 = st.columns([1, 1])
with sync_c1:
    if st.button("💾 Save all trades to Google Drive", use_container_width=True):
        if drive_service and save_to_drive(drive_service, st.session_state.trades):
            st.success("Saved.")
with sync_c2:
    if st.button("📥 Reload trades from Google Drive", use_container_width=True):
        if drive_service:
            st.session_state.trades = load_from_drive(drive_service) or []
            st.rerun()

st.markdown("---")

# ---------------- External Tools ----------------
st.subheader("External Tools")
t1, t2, t3, t4 = st.columns(4)
t1.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
t2.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
t3.link_button("Option Screener", "https://optionmoves.com/screener", use_container_width=True)
t4.link_button("IV Rank Check", "https://marketchameleon.com/", use_container_width=True)
