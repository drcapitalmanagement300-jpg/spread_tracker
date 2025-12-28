import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# Import persistence
from persistence import (
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    log_completed_trade, # <--- NEW IMPORT
    logout 
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Sniper: Dashboard")

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'

# --- AUTH FLOW ---
if "credentials" not in st.session_state:
    st.info("Please sign in to access your Dashboard.")
    # (Your existing auth button logic would go here if not already handled by main navigation)
    st.stop()

drive_service = build_drive_service_from_session()

# Load Active Trades
if "trades" not in st.session_state or not st.session_state.trades:
    st.session_state.trades = load_from_drive(drive_service) or []

# --- UI HEADER ---
col1, col2, col3 = st.columns([1.5, 7, 1.5])
with col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except:
        st.write("**DR CAPITAL**")
with col2:
    st.title("Active Positions")
with col3:
    if st.button("Log out"):
        logout()
        st.rerun()

st.markdown("---")

# --- ACTIVE TRADES DISPLAY ---
if not st.session_state.trades:
    st.info("No active trades. Go to 'Spread Finder' to add new positions.")
else:
    # Convert list to nice UI
    for i, trade in enumerate(st.session_state.trades):
        
        # Fetch current price for context (Optional but helpful)
        current_price = 0.0
        try:
            ticker_obj = yf.Ticker(trade['ticker'])
            current_price = ticker_obj.fast_info['last_price']
        except:
            pass

        with st.container():
            # Card Styling
            st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #00C853; margin-bottom: 15px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="margin:0; color:white;">{trade['ticker']} <span style="font-size:0.6em; color:#888;">{trade['contracts']}x Contracts</span></h2>
                        <p style="margin:0; color:#AAA;">Put Credit Spread • {trade['short_strike']}/{trade['long_strike']}</p>
                    </div>
                    <div style="text-align:right;">
                        <h3 style="margin:0; color:#00C853;">${trade['credit']:.2f} <span style="font-size:0.5em; color:#AAA;">CREDIT</span></h3>
                        <p style="margin:0; color:#AAA;">Current Stock: ${current_price:.2f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- ACTIONS (EXPANDER) ---
            with st.expander(f"Manage {trade['ticker']} Position"):
                
                c1, c2 = st.columns(2)
                
                # Column 1: Close Trade Logic
                with c1:
                    st.subheader("Log & Close Trade")
                    st.caption("Enter the debit paid (price per share) to buy back the spread.")
                    
                    debit_input = st.number_input(
                        f"Debit Paid ($)", 
                        min_value=0.0, 
                        max_value=100.0, 
                        step=0.01, 
                        format="%.2f",
                        key=f"debit_{i}"
                    )
                    
                    notes_input = st.text_input("Notes (Why did you close?)", key=f"notes_{i}")
                    
                    # Calculate P&L Preview
                    est_pl = (float(trade['credit']) - debit_input) * int(trade['contracts']) * 100
                    pl_color = SUCCESS_COLOR if est_pl > 0 else WARNING_COLOR
                    st.markdown(f"**Estimated P/L:** <span style='color:{pl_color}'>${est_pl:.2f}</span>", unsafe_allow_html=True)
                    
                    if st.button("✅ Confirm Log & Close", key=f"close_{i}"):
                        # 1. Prepare Data
                        trade_to_log = trade.copy()
                        trade_to_log['debit_paid'] = debit_input
                        trade_to_log['notes'] = notes_input
                        
                        # 2. Send to Google Sheet
                        with st.spinner("Logging to Cloud..."):
                            success = log_completed_trade(drive_service, trade_to_log)
                        
                        if success:
                            # 3. Remove from Active List
                            st.session_state.trades.pop(i)
                            save_to_drive(drive_service, st.session_state.trades)
                            st.success(f"Trade logged! Profit: ${est_pl:.2f}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to log to Google Sheet. Check connection.")

                # Column 2: Delete (Mistake) Logic
                with c2:
                    st.subheader("Delete (Mistake)")
                    st.caption("Removes trade from dashboard WITHOUT logging it to history.")
                    if st.button("🗑 Delete (Do not Log)", key=f"del_active_{i}"):
                        st.session_state.trades.pop(i)
                        save_to_drive(drive_service, st.session_state.trades)
                        st.rerun()
