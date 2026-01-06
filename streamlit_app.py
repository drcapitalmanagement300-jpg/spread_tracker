import streamlit as st
import pandas as pd
from datetime import datetime
from persistence import (
    build_drive_service_from_session, 
    log_new_trade, 
    get_trade_log, 
    update_trade_log,
    logout
)

st.set_page_config(page_title="Dashboard", layout="wide")

# --- SIDEBAR & AUTH ---
with st.sidebar:
    st.title("🦁 DR CAPITAL")
    st.write("Trading Dashboard")

# --- MAIN ENTRY FORM ---
st.title("New Trade Entry")

with st.form("entry_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ticker = st.text_input("Ticker (e.g. SPX)").upper()
        strategy = st.selectbox("Strategy", ["Iron Condor", "Credit Spread", "Debit Spread", "Naked Put", "Other"])
    with c2:
        entry_date = st.date_input("Entry Date", datetime.today())
        direction = st.selectbox("Direction", ["Neutral", "Bullish", "Bearish"])
    with c3:
        short_strike = st.number_input("Short Strike", value=0.0, step=5.0)
        long_strike = st.number_input("Long Strike", value=0.0, step=5.0)
    with c4:
        contracts = st.number_input("Contracts", min_value=1, value=1)
        credit = st.number_input("Entry Credit ($)", value=0.0, step=0.05)

    notes = st.text_area("Notes")
    earnings_date = st.date_input("Next Earnings Date", value=None)

    submitted = st.form_submit_button("Log Trade")

    if submitted:
        if not ticker:
            st.error("Ticker is required!")
        else:
            # Construct Trade Data Dictionary
            # Note: We specifically include 'Contracts' here
            trade_data = {
                'Ticker': ticker,
                'Entry_Date': str(entry_date),
                'Exit_Date': '', # Empty until closed
                'Strategy': strategy,
                'Direction': direction,
                'Short_Strike': short_strike,
                'Long_Strike': long_strike,
                'Contracts': contracts,
                'Credit': credit,
                'Debit': 0.0,
                'Realized_PL': 0.0,
                'Status': 'Open',
                'Notes': notes,
                'Earnings_Date': str(earnings_date) if earnings_date else ''
            }
            
            log_new_trade(trade_data)
            st.success(f"Logged {strategy} on {ticker}")
            st.rerun()

st.markdown("---")

# --- OPEN POSITIONS MANAGEMENT ---
st.subheader("Manage Open Positions")

logs = get_trade_log()
# Filter only for Open trades
open_trades = [t for t in logs if t.get('Status') == 'Open']

if not open_trades:
    st.info("No open positions.")
else:
    for i, trade in enumerate(open_trades):
        # We need to find the REAL index in the master list to update the correct row
        # (Simple approach: we iterate the full list in get_trade_log, but here we just need to be careful.
        # Ideally, rows should have IDs. For now, we assume the list index matches persistence.)
        # To be safe, we will pass the index relative to the loaded dataframe if using the pandas method in persistence.
        
        # Let's create a visual card for the trade
        with st.expander(f"{trade['Ticker']} - {trade['Strategy']} ({trade['Entry_Date']})"):
            c_info, c_close = st.columns([3, 1])
            
            with c_info:
                st.write(f"**Strikes:** {trade['Short_Strike']}/{trade['Long_Strike']}")
                st.write(f"**Contracts:** {trade.get('Contracts', 1)}")
                st.write(f"**Original Credit:** ${trade.get('Credit', 0)}")
            
            with c_close:
                st.write("**Close Trade**")
                close_price = st.number_input("Closing Debit ($)", min_value=0.0, step=0.01, key=f"cp_{i}")
                
                if st.button("Close Position", key=f"btn_close_{i}"):
                    # 1. CALC P&L
                    # P&L = (Entry Credit - Exit Debit) * Contracts * 100
                    try:
                        entry_credit = float(trade.get('Credit', 0))
                        num_contracts = float(trade.get('Contracts', 1))
                        pnl = (entry_credit - close_price) * num_contracts * 100
                    except:
                        pnl = 0
                    
                    # 2. GET DATE
                    exit_date_str = datetime.now().strftime("%Y-%m-%d")
                    
                    # 3. UPDATE LOG
                    # We need the index in the original file. 
                    # If 'open_trades' is a subset, 'i' is wrong.
                    # FIX: We find the index in the original 'logs' list
                    original_index = logs.index(trade)
                    
                    updates = {
                        "Exit_Date": exit_date_str,
                        "Debit": close_price,
                        "Realized_PL": pnl,
                        "Status": "Closed"
                    }
                    
                    update_trade_log(None, original_index, updates)
                    st.success(f"Trade Closed! P&L: ${pnl:.2f}")
                    st.rerun()
