import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

# Import persistence
from persistence import (
    build_drive_service_from_session,
    get_trade_log,
    delete_log_entry,
    logout
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Sniper: Journal")

# --- CONSTANTS & STYLING ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#757575"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'

# Configure Matplotlib for Dark Theme
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": "#444444",
    "grid.alpha": 0.3
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

# --- HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except:
        st.write("**DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Trade Journal</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Performance Analytics & Post-Mortem</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.write("") 

st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- FETCH DATA ---
raw_logs = get_trade_log(drive_service)

if not raw_logs:
    st.info("No closed trades found in the Journal yet.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(raw_logs)

# Data Cleaning
try:
    df['Realized_PL'] = pd.to_numeric(df['Realized_PL'], errors='coerce').fillna(0.0)
    df['Contracts'] = pd.to_numeric(df['Contracts'], errors='coerce').fillna(1)
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0.0)
except Exception as e:
    st.error(f"Data formatting error: {e}")
    st.stop()

# --- ANALYTICS ENGINE ---
total_trades = len(df)
wins = df[df['Realized_PL'] > 0]
losses = df[df['Realized_PL'] <= 0]

total_pl = df['Realized_PL'].sum()
win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0

avg_win = wins['Realized_PL'].mean() if not wins.empty else 0
avg_loss = losses['Realized_PL'].mean() if not losses.empty else 0

gross_profit = wins['Realized_PL'].sum()
gross_loss = abs(losses['Realized_PL'].sum())
profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

# --- METRICS ROW ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net P&L", f"${total_pl:,.2f}", delta_color="normal")
m2.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W - {len(losses)}L")
m3.metric("Profit Factor", f"{profit_factor:.2f}", help="> 1.5 is healthy")
m4.metric("Avg Win", f"${avg_win:.2f}")
m5.metric("Avg Loss", f"${avg_loss:.2f}", delta_color="inverse")

st.markdown("---")

# --- CHARTS (Matplotlib) ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Equity Curve")
    if not df.empty:
        df_sorted = df.sort_values(by="Exit_Date")
        df_sorted['Cumulative_PL'] = df_sorted['Realized_PL'].cumsum()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot Line
        ax.plot(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR, linewidth=2, marker='o', markersize=4)
        
        # Fill Area
        ax.fill_between(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR, alpha=0.1)
        
        # Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_ylabel("Total P&L ($)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig, use_container_width=True)
    else:
        st.caption("Not enough data for chart.")

with c2:
    st.subheader("Win/Loss Ratio")
    if total_trades > 0:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        
        sizes = [len(wins), len(losses)]
        labels = ['Wins', 'Losses']
        colors = [SUCCESS_COLOR, WARNING_COLOR]
        
        # Donut Chart
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                           startangle=90, pctdistance=0.85, textprops={'color':"white"})
        
        # Draw Circle for Donut
        centre_circle = plt.Circle((0,0),0.70,fc=BG_COLOR)
        fig2.gca().add_artist(centre_circle)
        
        ax2.axis('equal')  
        st.pyplot(fig2, use_container_width=True)
    else:
        st.caption("No trades.")

# --- TICKER PERFORMANCE (Matplotlib Bar) ---
st.subheader("Performance by Ticker")
if not df.empty:
    ticker_perf = df.groupby('Ticker')['Realized_PL'].sum().reset_index()
    ticker_perf = ticker_perf.sort_values(by='Realized_PL', ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    
    # Color mapping
    bar_colors = [SUCCESS_COLOR if x > 0 else WARNING_COLOR for x in ticker_perf['Realized_PL']]
    
    ax3.bar(ticker_perf['Ticker'], ticker_perf['Realized_PL'], color=bar_colors, alpha=0.9)
    
    ax3.set_ylabel("Net P&L ($)")
    ax3.axhline(0, color='gray', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    st.pyplot(fig3, use_container_width=True)

st.markdown("---")

# --- THE GALLERY (Trade Cards) ---
st.subheader("Trade History")

# Reverse order to show newest first
for i, row in df.iloc[::-1].iterrows():
    
    pl = row['Realized_PL']
    is_win = pl > 0
    border_color = SUCCESS_COLOR if is_win else WARNING_COLOR
    card_bg = "rgba(0, 200, 83, 0.05)" if is_win else "rgba(211, 47, 47, 0.05)"
    
    with st.container():
        st.markdown(f"""
        <div style="border-left: 5px solid {border_color}; background-color: {card_bg}; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:white;">{row['Ticker']} <span style="font-size:0.7em; color:#888;">{row['Exit_Date'].strftime('%b %d')}</span></h3>
                <h3 style="margin:0; color:{border_color};">${pl:,.2f}</h3>
            </div>
            <div style="display:flex; gap: 20px; margin-top: 10px; font-size: 14px; color: #ddd;">
                <span><strong>Strikes:</strong> {row['Short_Strike']}/{row['Long_Strike']}</span>
                <span><strong>Credit:</strong> ${row['Credit']}</span>
                <span><strong>Close Debit:</strong> ${row.get('Debit_Paid', '0.00')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Expander for details and actions
        with st.expander(f"Details & Notes for {row['Ticker']}"):
            c_note, c_act = st.columns([3, 1])
            with c_note:
                st.write(f"**Notes:** {row.get('Notes', 'No notes.')}")
                st.caption(f"Trade ID: {row.get('Trade_ID', 'N/A')}")
            with c_act:
                # Delete Button
                # Note: We pass 'i' which is the DataFrame index. 
                # This ensures we are targeting the correct row in the original list.
                if st.button("🗑 Delete Log", key=f"del_{i}"):
                    if delete_log_entry(drive_service, i):
                        st.success("Deleted. Refreshing...")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")
