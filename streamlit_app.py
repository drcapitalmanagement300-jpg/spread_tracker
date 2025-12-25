#streamlit_app.py Dec 24 2025 11:18

import streamlit as st
from datetime import date, datetime
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh
import io
import html
def esc(x):
    return html.escape(str(x)) if x is not None else ""

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

# Try to import Google API helpers for the journal feature
try:
    from googleapiclient.http import MediaIoBaseUpload
except ImportError:
    MediaIoBaseUpload = None

# ---------------- Page config ----------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")

# ---------------- UI Refresh ----------------
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth configured.")

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
        return 30 # fallback

# ---------------- Journal Helper ----------------
def append_to_drive_journal(service, trade_data, notes):
    """
    Appends a log entry to 'trading_journal.txt' in the root of the Drive.
    Includes PnL and Breached Conditions.
    """
    if not service or not MediaIoBaseUpload:
        return False

    # Extract cached data for logging
    cached = trade_data.get("cached", {})
    profit_pct = cached.get("current_profit_percent")
    
    # Determine Profit/Loss String
    if profit_pct is not None:
        pnl_str = f"{profit_pct:.2f}%"
    else:
        pnl_str = "N/A (Data missing)"

    # Determine Breached Conditions
    conditions = []
    rules = cached.get("rule_violations", {})
    abs_delta = cached.get("abs_delta")
    spread_val = cached.get("spread_value_percent")
    current_dte = days_to_expiry(trade_data["expiration"])

    if profit_pct and profit_pct >= 50:
        conditions.append("PROFIT TARGET REACHED")
    
    if rules.get("other_rules", False):
        if abs_delta and abs_delta >= 0.40:
            conditions.append(f"DELTA BREACH ({abs_delta:.2f} >= 0.40)")
        if spread_val and spread_val >= 150:
            conditions.append(f"SPREAD VALUE BREACH ({spread_val:.0f}% >= 150%)")
        if current_dte <= 7:
             conditions.append(f"DTE LOW ({current_dte} <= 7)")

    if not conditions:
        conditions.append("None (Manual Close)")

    filename = "trading_journal.txt"
    log_entry = (
        f"\n{'='*30}\n"
        f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"ACTION: CLOSED POSITION\n"
        f"TICKER: {trade_data.get('ticker')}\n"
        f"STRIKES: -{trade_data.get('short_strike')} / +{trade_data.get('long_strike')}\n"
        f"EXPIRY: {trade_data.get('expiration')}\n"
        f"FINAL P/L: {pnl_str}\n"
        f"CONDITIONS: {', '.join(conditions)}\n"
        f"NOTES: {notes}\n"
        f"{'='*30}\n"
    )

    try:
        # 1. Search for existing file
        query = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            # Create new file
            file_metadata = {'name': filename, 'mimeType': 'text/plain'}
            media = MediaIoBaseUpload(io.BytesIO(log_entry.encode('utf-8')), mimetype='text/plain', resumable=True)
            service.files().create(body=file_metadata, media_body=media).execute()
        else:
            # Update existing file
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)
            current_content = request.execute()
            new_content = current_content + log_entry.encode('utf-8')
            media = MediaIoBaseUpload(io.BytesIO(new_content), mimetype='text/plain', resumable=True)
            service.files().update(fileId=file_id, media_body=media).execute()
        return True
    except Exception as e:
        print(f"Journal Error: {e}")
        return False

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
            st.session_state.trades.append(trade)
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
        cached = t.get("cached", {})

        current_dte = days_to_expiry(t["expiration"])
        entry_dte = get_entry_dte(t["entry_date"], t["expiration"])
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"]
        max_loss = width - t["credit"]

        # Backend Data
        current_price = cached.get("current_price")
        # Attempt to get daily change. Default to 0.0 if missing.
        daily_change_pct = cached.get("daily_change_percent", 0.0) 

        abs_delta = cached.get("abs_delta")
        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        # --- Price Display Logic ---
        if current_price is not None:
            price_display = f"${current_price:.2f}"
            
            # Change logic (Arrow and Color)
            if daily_change_pct is None: 
                daily_change_pct = 0.0
            
            if daily_change_pct >= 0:
                change_color = "green"
                change_arrow = "▲"
                change_display = f"{daily_change_pct:.2f}%"
            else:
                change_color = "#d32f2f" # Red
                change_arrow = "▼"
                change_display = f"{abs(daily_change_pct):.2f}%"
                
            price_html = f"""
                <span style="font-size: 16px; font-weight: 500; color: #333; margin-left: 8px;">{price_display}</span>
                <span style="font-size: 14px; color: {change_color}; margin-left: 6px;">{change_arrow} {change_display}</span>
            """
        else:
            price_html = "<span style='font-size: 14px; color: gray; margin-left: 8px;'>Loading...</span>"


        # --- Status Logic ---
        status_msg = "Status Nominal"
        status_icon = "✅"
        status_color = "green"

        if rules.get("other_rules", False):
            status_icon = "⚠️"
            status_color = "#d32f2f" # Red
            if abs_delta and abs_delta >= 0.40:
                status_msg = "Short Delta High"
            elif spread_value and spread_value >= 150:
                status_msg = "Spread Value High"
            elif current_dte <= 7:
                status_msg = "Expiration Imminent"
        
        if profit_pct and profit_pct >= 50:
            status_icon = "💰" 
            status_msg = "Profit Target Reached"
            status_color = "green"

        # --- Color Coding ---
        delta_color = "#d32f2f" if abs_delta and abs_delta >= 0.40 else "green"
        delta_val = f"{abs_delta:.2f}" if abs_delta is not None else "Pending"

        spread_color = "#d32f2f" if spread_value and spread_value >= 150 else "green"
        spread_val = f"{spread_value:.0f}" if spread_value is not None else "Pending"

        dte_color = "#d32f2f" if current_dte <= 7 else "green"

        if profit_pct is None:
            profit_color = "inherit"
            profit_val = "Pending"
        else:
            profit_val = f"{profit_pct:.1f}"
            if profit_pct >= 50:
                profit_color = "green"
            elif profit_pct < 0:
                profit_color = "#d32f2f"
            else:
                profit_color = "#e6b800"

        cols = st.columns([3, 4])

        # -------- LEFT CARD (Details + Close Button) --------
with cols[0]:
    st.markdown(
        f"""
        <div style="line-height: 1.4; font-size: 15px;">
            <div style="display: flex; align-items: baseline; margin-bottom: 8px;">
                <h3 style="margin: 0; padding: 0;">{esc(t['ticker'])}</h3>
                {price_html}
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2px;">
                <div><strong>Short:</strong> {esc(t['short_strike'])}</div>
                <div><strong>Max Gain:</strong> {esc(format_money(max_gain))}</div>
                <div><strong>Long:</strong> {esc(t['long_strike'])}</div>
                <div><strong>Max Loss:</strong> {esc(format_money(max_loss))}</div>
                <div style="grid-column: span 2;">
                    <strong>Exp:</strong> {esc(t['expiration'])}
                </div>
                <div style="grid-column: span 2;">
                    <strong>Width:</strong> {esc(f"{width:.2f}")}
                </div>
            </div>

            <div style="
                margin-top: 15px;
                padding-top: 10px;
                border-top: 1px solid #eee;
                color: {status_color};
                font-weight: bold;
            ">
                {esc(status_icon)} {esc(status_msg)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
            st.write("") # Spacer
            
            # --- Close / Log Logic ---
            if st.button("Close Position / Log", key=f"btn_close_{i}"):
                st.session_state[f"close_mode_{i}"] = True

            if st.session_state.get(f"close_mode_{i}", False):
                with st.container():
                    st.markdown("---")
                    st.info("Log entry for Trading Journal")
                    with st.form(key=f"close_form_{i}"):
                        close_notes = st.text_area("Closing Notes / Reason", height=80)
                        submit_close = st.form_submit_button("Confirm Close & Log")
                        
                        if submit_close:
                            if drive_service:
                                saved_journal = append_to_drive_journal(drive_service, t, close_notes)
                                if saved_journal:
                                    st.success("Entry added to 'trading_journal.txt'")
                                else:
                                    st.error("Could not write to Drive journal.")
                            
                            st.session_state.trades.pop(i)
                            if drive_service:
                                save_to_drive(drive_service, st.session_state.trades)
                            
                            del st.session_state[f"close_mode_{i}"]
                            st.experimental_rerun()

                    if st.button("Cancel", key=f"cancel_{i}"):
                        del st.session_state[f"close_mode_{i}"]
                        st.experimental_rerun()

        # -------- RIGHT CARD (Alerts & Chart) --------
        with cols[1]:
            st.markdown(
                f"""
                <div style="font-size: 14px; margin-bottom: 10px;">
                    <div>Short-delta: <strong style='color:{delta_color}'>{delta_val}</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not exceed 0.40)</span></div>
                    <div>Spread Value: <strong style='color:{spread_color}'>{spread_val}%</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not exceed 150%)</span></div>
                    <div>DTE: <strong style='color:{dte_color}'>{current_dte}</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not be less than 7)</span></div>
                    <div>Profit: <strong style='color:{profit_color}'>{profit_val}%</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must sell between 50-75%)</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- CHART LOGIC (DTE Axis) ---
            if t.get("pnl_history"):
                df = pd.DataFrame(t["pnl_history"])
                
                base = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X(
                        "dte:Q", 
                        title="Days to Expiration (DTE)", 
                        scale=alt.Scale(domain=[entry_dte, 0])
                    ),
                    y=alt.Y("profit:Q", scale=alt.Scale(domain=[-100, 100]), title="Profit %"),
                    tooltip=["dte", "profit", "date"]
                ).properties(height=200)

                line_50 = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y")
                line_75 = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(color="green", strokeDash=[5,5]).encode(y="y")
                line_0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeWidth=1).encode(y="y")

                st.altair_chart(base + line_50 + line_75 + line_0, use_container_width=True)
            else:
                st.caption("Waiting for market data history...")

        st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# ---------------- Manual Controls ----------------
st.write("### Data Sync")
ctl1, ctl2, ctl_spacer = st.columns([1.5, 1.5, 3.5])

with ctl1:
    if st.button("💾 Save all trades to Google Drive"):
        saved = False
        if drive_service:
            try:
                saved = save_to_drive(drive_service, st.session_state.trades)
            except Exception:
                saved = False
        if saved:
            st.success("Saved successfully.")
        else:
            st.error("Failed to save.")

with ctl2:
    if st.button("📥 Reload trades from Google Drive"):
        if drive_service:
            loaded = load_from_drive(drive_service)
            if loaded is not None:
                st.session_state.trades = loaded
                st.success("Loaded trades.")
                st.experimental_rerun()
            else:
                st.info("No trades found/failed.")

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
