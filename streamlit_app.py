import streamlit as st
from datetime import date, datetime, timezone
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh
import io
import json

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

try:
    from googleapiclient.http import MediaIoBaseUpload
except ImportError:
    MediaIoBaseUpload = None

# ---------------- Page config ----------------
st.set_page_config(page_title="DR Capital Spread Monitor", layout="wide")

# ---------------- UI Refresh ----------------
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth configuration required.")

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

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

# ---------------- Journal Helper ----------------
def append_to_drive_journal(service, trade_data, notes):
    if not service or not MediaIoBaseUpload:
        return False

    cached = trade_data.get("cached", {})
    profit_pct = cached.get("current_profit_percent")
    pnl_str = f"{profit_pct:.2f}%" if profit_pct is not None else "N/A"

    conditions = []
    rules = cached.get("rule_violations", {})
    abs_delta = cached.get("abs_delta")
    spread_val = cached.get("spread_value_percent")
    current_dte = days_to_expiry(trade_data["expiration"])

    if profit_pct and profit_pct >= 50:
        conditions.append("PROFIT TARGET REACHED")
    if rules.get("other_rules", False):
        if abs_delta and abs_delta >= 0.40: conditions.append(f"DELTA BREACH ({abs_delta:.2f})")
        if spread_val and spread_val >= 150: conditions.append(f"SPREAD VALUE BREACH ({spread_val:.0f}%)")
        if current_dte <= 7: conditions.append(f"DTE LOW ({current_dte})")

    filename = "trading_journal.txt"
    log_entry = (
        f"\n{'='*30}\n"
        f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"TICKER: {trade_data.get('ticker')}\n"
        f"FINAL P/L: {pnl_str}\n"
        f"CONDITIONS: {', '.join(conditions) if conditions else 'Manual Close'}\n"
        f"NOTES: {notes}\n"
        f"{'='*30}\n"
    )

    try:
        query = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            file_metadata = {'name': filename, 'mimeType': 'text/plain'}
            media = MediaIoBaseUpload(io.BytesIO(log_entry.encode('utf-8')), mimetype='text/plain')
            service.files().create(body=file_metadata, media_body=media).execute()
        else:
            file_id = files[0]['id']
            current_content = service.files().get_media(fileId=file_id).execute()
            new_content = current_content + log_entry.encode('utf-8')
            media = MediaIoBaseUpload(io.BytesIO(new_content), mimetype='text/plain')
            service.files().update(fileId=file_id, media_body=media).execute()
        return True
    except Exception as e:
        return False

# ---------------- Header ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    st.write("### DR CAPITAL")
with header_col2:
    st.markdown("<h1>Put Credit Spread Monitor</h1><p style='color: gray;'>Strategic Options Management System</p>", unsafe_allow_html=True)
with header_col3:
    if st.button("Log out"):
        logout()
        st.experimental_rerun()

st.markdown("---")

# ---------------- Load State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
elif "trades" not in st.session_state:
    st.session_state.trades = []

# ---------------- Add Trade ----------------
with st.expander("➕ Open New Position"):
    with st.form("add_trade", clear_on_submit=True):
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

        if st.form_submit_button("Initialize Position"):
            if ticker and short_strike > long_strike:
                trade = {
                    "id": f"{ticker}-{datetime.now().timestamp()}",
                    "ticker": ticker, "short_strike": short_strike, "long_strike": long_strike,
                    "expiration": expiration.isoformat(), "credit": credit, "entry_date": entry_date.isoformat(),
                    "cached": {}, "pnl_history": []
                }
                st.session_state.trades.append(trade)
                if drive_service: save_to_drive(drive_service, st.session_state.trades)
                st.experimental_rerun()

# ---------------- Display Trades ----------------
for i, t in enumerate(st.session_state.trades):
    cached = t.get("cached", {})
    current_dte = days_to_expiry(t["expiration"])
    entry_dte = get_entry_dte(t["entry_date"], t["expiration"])
    width = abs(t["short_strike"] - t["long_strike"])
    
    price = cached.get("current_price", 0.0)
    change = cached.get("daily_change_percent", 0.0)
    delta = cached.get("abs_delta")
    spread_val = cached.get("spread_value_percent")
    profit_pct = cached.get("current_profit_percent")
    rules = cached.get("rule_violations", {})

    status_icon, status_msg, status_color = ("✅", "Status Nominal", "green")
    if rules.get("other_rules"):
        status_icon, status_color = ("⚠️", "#d32f2f")
        if delta and delta >= 0.40: status_msg = "Delta Breach"
        elif spread_val and spread_val >= 150: status_msg = "Spread High"
        elif current_dte <= 7: status_msg = "Low DTE"
    if profit_pct and profit_pct >= 50:
        status_icon, status_msg, status_color = ("💰", "Target Reached", "green")

    # Layout Container
    st.markdown(f"""<div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px;">""", unsafe_allow_html=True)
    
    cols = st.columns([4, 6])
    
    with cols[0]:
        c_arrow = "▲" if change >= 0 else "▼"
        c_color = "green" if change >= 0 else "#d32f2f"
        
        # This fixes the raw HTML issue by ensuring Markdown handles the render
        st.markdown(f"""
            <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px;">
                <h2 style="margin:0;">{t['ticker']}</h2>
                <span style="font-size: 20px; font-weight: bold;">${price:.2f}</span>
                <span style="color: {c_color}; font-size: 16px;">{c_arrow} {abs(change):.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        d1.write(f"**Short:** {t['short_strike']}")
        d1.write(f"**Long:** {t['long_strike']}")
        d1.write(f"**Exp:** {t['expiration']}")
        d2.write(f"**Gain:** {format_money(t['credit'])}")
        d2.write(f"**Loss:** {format_money(width - t['credit'])}")
        d2.write(f"**Width:** {width:.2f}")

        st.markdown(f"""
            <div style="background: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 13px; border-left: 4px solid {status_color};">
                <div>Delta: <b style="color:{'red' if delta and delta >= 0.4 else 'green'}">{f"{delta:.2f}" if delta else 'Pending'}</b></div>
                <div>Spread: <b style="color:{'red' if spread_val and spread_val >= 150 else 'green'}">{f"{spread_val:.0f}%" if spread_val else 'Pending'}</b></div>
                <div>DTE: <b style="color:{'red' if current_dte <= 7 else 'green'}">{current_dte}</b></div>
                <div>Profit: <b style="color:{status_color}">{f"{profit_pct:.1f}%" if profit_pct else 'Pending'}</b></div>
            </div>
            <div style="margin-top: 10px; font-weight: bold; color: {status_color};">{status_icon} {status_msg}</div>
        """, unsafe_allow_html=True)

        if st.button("Close Position", key=f"btn_{i}", use_container_width=True):
            st.session_state[f"m_{i}"] = True

    with cols[1]:
        if t.get("pnl_history"):
            df = pd.DataFrame(t["pnl_history"])
            line = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("dte:Q", title="DTE", scale=alt.Scale(domain=[entry_dte, 0])),
                y=alt.Y("profit:Q", title="Profit %", scale=alt.Scale(domain=[-100, 100])),
                tooltip=["date", "dte", "profit"]
            ).properties(height=250)
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("Performance history pending backend update...")

    if st.session_state.get(f"m_{i}"):
        with st.form(f"form_{i}"):
            notes = st.text_area("Notes")
            if st.form_submit_button("Confirm Close"):
                if drive_service: append_to_drive_journal(drive_service, t, notes)
                st.session_state.trades.pop(i)
                if drive_service: save_to_drive(drive_service, st.session_state.trades)
                st.experimental_rerun()
        if st.button("Cancel", key=f"can_{i}"):
            del st.session_state[f"m_{i}"]
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.sidebar.subheader("System Sync")
if st.sidebar.button("💾 Save Trades"):
    if drive_service: save_to_drive(drive_service, st.session_state.trades)
if st.sidebar.button("📥 Reload Trades"):
    st.experimental_rerun()

st.subheader("External Tools")
t1, t2, t3, t4 = st.columns(4)
t1.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
t2.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
t3.link_button("Option Screener", "https://optionmoves.com/screener", use_container_width=True)
t4.link_button("Market Chameleon", "https://marketchameleon.com/", use_container_width=True)
