import streamlit as st
from datetime import date, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from streamlit_autorefresh import st_autorefresh

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    log_completed_trade, 
    logout,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="Dashboard", layout="wide")

# ---------------- Constants ----------------
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
STOP_LOSS_COLOR = "#FFA726"
WHITE_DIVIDER_HTML = "<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>"

# ---------------- UI Refresh ----------------
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
ensure_logged_in()

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# ---------------- Header ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])

with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
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
        st.rerun()

st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

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

# --- Charting & Progress Bar ---
def render_profit_bar(profit_pct):
    if profit_pct is None:
        return '<div style="color:gray; font-size:12px;">Pending P&L...</div>'
    
    fill_pct = ((profit_pct + 100) / 175) * 100 
    display_fill = max(0, min(fill_pct, 100))
    
    if profit_pct < 0:
        bar_color = WARNING_COLOR 
        status_text = f"LOSS: {profit_pct:.1f}%"
    else:
        bar_color = SUCCESS_COLOR
        status_text = f"PROFIT: {profit_pct:.1f}%"

    return (
        f'<div style="margin-bottom: 12px; margin-top: 5px;">'
        f'<div style="width: 100%; background-color: #333; height: 6px; border-radius: 3px; position: relative; overflow: hidden; border: 1px solid #444;">'
        f'<div style="width: {display_fill}%; background-color: {bar_color}; height: 100%;"></div>'
        f'</div></div>'
    )

# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Display Trades ----------------
if not st.session_state.trades:
    st.info("No active trades. Go to 'Spread Finder' to scan for new opportunities.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})

        current_dte = days_to_expiry(t["expiration"])
        
        # --- FIXED: SAFE CONTRACTS ACCESS ---
        contracts = t.get("contracts", 1) 
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain_total = t["credit"] * 100 * contracts
        max_loss_total = (width - t["credit"]) * 100 * contracts

        current_price = cached.get("current_price")
        profit_pct = cached.get("current_profit_percent")
        
        cols = st.columns([3, 4])

        # -------- LEFT CARD --------
        with cols[0]:
            st.markdown(f"### {t['ticker']} <span style='font-size:0.7em'>${current_price if current_price else '-.--'}</span>", unsafe_allow_html=True)
            st.write(f"Strikes: {t['short_strike']}/{t['long_strike']}")
            st.write(f"Contracts: **{contracts}**")
            
            if st.button("Close Position / Log", key=f"btn_close_{i}"):
                st.session_state[f"close_mode_{i}"] = True

            if st.session_state.get(f"close_mode_{i}", False):
                with st.container():
                    st.markdown("---")
                    st.info("📉 Closing Position & Logging to Journal")
                    with st.form(key=f"close_form_{i}"):
                        col_log1, col_log2 = st.columns(2)
                        with col_log1:
                            debit_paid = st.number_input("Debit Paid ($)", min_value=0.0, step=0.01)
                        with col_log2:
                            close_notes = st.text_area("Notes", height=70)
                        
                        if st.form_submit_button("Confirm Close"):
                            if drive_service:
                                trade_data = t.copy()
                                
                                # --- FIXED: DATA MAPPING FOR LOG ---
                                trade_data['debit_paid'] = debit_paid
                                trade_data['notes'] = close_notes
                                trade_data['exit_date'] = str(date.today()) # Auto-set date
                                trade_data['contracts'] = contracts         # Pass contracts
                                
                                if log_completed_trade(drive_service, trade_data):
                                    st.success(f"Logged {t['ticker']}")
                                    st.session_state.trades.pop(i)
                                    save_to_drive(drive_service, st.session_state.trades)
                                    del st.session_state[f"close_mode_{i}"]
                                    st.rerun()
                                else:
                                    st.error("Drive Error: Could not log to sheet.")
                            else:
                                st.error("No Drive Service Connected.")

                    if st.button("Cancel", key=f"cancel_{i}"):
                        del st.session_state[f"close_mode_{i}"]
                        st.rerun()

        # -------- RIGHT CARD --------
        with cols[1]:
            st.markdown(render_profit_bar(profit_pct), unsafe_allow_html=True)
            st.caption(f"Max Gain: {format_money(max_gain_total)} | DTE: {current_dte}")

        st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

# ---------------- Footer ----------------
t1, t2 = st.columns(2)
with t1: st.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
with t2: st.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
