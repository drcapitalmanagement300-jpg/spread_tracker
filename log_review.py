import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import textwrap
import os
from datetime import datetime

# Import persistence
from persistence import (
    build_drive_service_from_session,
    get_trade_log,
    delete_log_entry,
    logout
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Log Review")

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#757575"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#444444", "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "axes.grid": True, "grid.color": "#444444", "grid.alpha": 0.3
})

# --- AUTH CHECK ---
drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    pass

if not drive_service:
    st.warning("Please sign in on the Dashboard page to view your Journal.")
    st.stop()

st.title("🦁 Trade Journal")

# --- FETCH DATA ---
raw_logs = get_trade_log(drive_service)

if not raw_logs:
    st.info("No closed trades found in the Journal yet.")
    st.stop()

df = pd.DataFrame(raw_logs)

# --- DATA CLEANING ---
try:
    # 1. Numeric Conversion
    cols_to_num = ['Realized_PL', 'Contracts', 'Credit', 'Short_Strike', 'Long_Strike', 'Debit']
    for c in cols_to_num:
        if c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    # 2. Date Parsing
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'], errors='coerce')
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Exit_Date', 'Entry_Date'])

    # 3. Duration & Logic
    def fix_year_rollover(row):
        if row['Entry_Date'] > row['Exit_Date']:
            return row['Entry_Date'] - pd.DateOffset(years=1)
        return row['Entry_Date']

    if not df.empty:
        df['Entry_Date'] = df.apply(fix_year_rollover, axis=1)
        df['Duration'] = (df['Exit_Date'] - df['Entry_Date']).dt.days
        df['Duration'] = df['Duration'].clip(lower=1)

except Exception as e:
    st.error(f"Data formatting error: {e}")
    st.dataframe(df.head())
    st.stop()

# --- METRICS ---
total_pl = df['Realized_PL'].sum()
win_rate = (len(df[df['Realized_PL'] > 0]) / len(df) * 100) if len(df) > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Net P&L", f"${total_pl:,.2f}")
m2.metric("Win Rate", f"{win_rate:.0f}%")
m3.metric("Trades", len(df))

# --- CHARTS ---
c1, c2 = st.columns(2)
df_sorted = df.sort_values(by="Exit_Date")

with c1:
    st.markdown("**Equity Curve**")
    if not df.empty:
        df_sorted['Cumulative_PL'] = df_sorted['Realized_PL'].cumsum()
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR)
        ax.fill_between(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR, alpha=0.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)

st.markdown("---")

# --- HISTORY ---
st.subheader("Trade History")
for i, row in df.iloc[::-1].iterrows():
    pl = row['Realized_PL']
    color = SUCCESS_COLOR if pl > 0 else WARNING_COLOR
    
    card_html = textwrap.dedent(f"""
        <div style="border-left: 4px solid {color}; background-color: rgba(255,255,255,0.05); padding: 10px; margin-bottom: 8px;">
            <div style="display:flex; justify-content:space-between;">
                <strong>{row['Ticker']}</strong>
                <span style="color:{color}">${pl:,.2f}</span>
            </div>
            <div style="font-size:12px; color:#aaa;">
                {row['Exit_Date'].strftime('%Y-%m-%d')} | Contracts: {int(row['Contracts'])}
            </div>
        </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)
    
    with st.expander("Details"):
        st.write(f"Notes: {row.get('Notes', '-')}")
        if st.button("Delete", key=f"del_{i}"):
            if delete_log_entry(drive_service, i):
                st.rerun()
