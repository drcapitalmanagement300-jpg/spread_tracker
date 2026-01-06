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

# --- CONSTANTS & STYLING ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#757575"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'
SAVED_COLOR = '#FFA726'
LOGO_FILE = "754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG"

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

# --- COMPACT CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 20px !important; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    
    .trade-card { padding: 10px 15px !important; margin-bottom: 8px !important; }
    .trade-card h3 { font-size: 16px !important; }
    .trade-details { font-size: 12px !important; margin-top: 5px !important; }
    
    .risk-badge { font-size: 11px; margin-top: 4px; color: #FFA726; font-weight: 500; }
    .earnings-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin-left: 10px; vertical-align: middle; }
    .earnings-danger { background-color: rgba(211, 47, 47, 0.2); color: #ff5252; border: 1px solid #ff5252; }
    .earnings-safe { background-color: rgba(0, 200, 83, 0.1); color: #69f0ae; border: 1px solid #00c853; }
</style>
""", unsafe_allow_html=True)

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
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=110)
    else:
        st.write("🦁 **DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 5px;'>
        <h2 style='margin-bottom: 0px; padding-bottom: 0px;'>Trade Journal</h2>
        <p style='margin-top: 0px; font-size: 14px; color: gray;'>Performance Analytics & Efficiency Audit</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.write("") 

# --- FETCH DATA ---
raw_logs = get_trade_log(drive_service)

if not raw_logs:
    st.info("No closed trades found in the Journal yet.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(raw_logs)

# --- DATA CLEANING & LOGIC FIXES ---
try:
    # 1. Robust Numeric Conversion
    cols_to_num = ['Realized_PL', 'Contracts', 'Credit', 'Short_Strike', 'Long_Strike']
    for c in cols_to_num:
        if c in df.columns:
            # Clean strings (remove $ and ,) safely by ensuring string type first
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[c] = df[c].replace('', '0')
            # Coerce to numeric (bad data becomes NaN)
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    # 2. Robust Date Parsing (The Fix for "0.93" error)
    # Convert to string first so "0.93" becomes text, which to_datetime treats as invalid (NaT)
    # instead of treating it as a float timestamp.
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'].astype(str), errors='coerce')
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'].astype(str), errors='coerce')
    
    # CRITICAL: Drop rows where dates failed to parse (Garbage data)
    df = df.dropna(subset=['Exit_Date', 'Entry_Date'])

    # Handle Earnings Date
    if 'Earnings_Date' not in df.columns:
        df['Earnings_Date'] = pd.NaT
    else:
        df['Earnings_Date'] = pd.to_datetime(df['Earnings_Date'].astype(str), errors='coerce')

    # 3. Handle Year Rollover (e.g. Dec 30 to Jan 2)
    def fix_year_rollover(row):
        # Safe to compare because we dropped NaNs above
        if row['Entry_Date'] > row['Exit_Date']:
            return row['Entry_Date'] - pd.DateOffset(years=1)
        return row['Entry_Date']

    if not df.empty:
        df['Entry_Date'] = df.apply(fix_year_rollover, axis=1)

        # 4. Duration Calculation
        df['Duration'] = (df['Exit_Date'] - df['Entry_Date']).dt.days
        df['Duration'] = df['Duration'].clip(lower=1)
        
        # 5. Max Loss & Risk Saved
        df['Spread_Width'] = (df['Short_Strike'] - df['Long_Strike']).abs()
        df['Max_Loss_Trade'] = (df['Spread_Width'] - df['Credit']) * 100 * df['Contracts']
        
        # Risk Saved Logic: Only applies to NEGATIVE P&L
        df['Risk_Saved'] = df.apply(lambda x: (x['Max_Loss_Trade'] - abs(x['Realized_PL'])) if x['Realized_PL'] < 0 else 0, axis=1)
    
    # Final check if DF became empty after cleaning
    if df.empty:
        st.warning("All log entries contained invalid data (dates). Please check the Google Sheet.")
        st.stop()

except Exception as e:
    st.error(f"Data formatting error: {e}")
    st.write("Debug Data:", df.head())
    st.stop()

# --- ANALYTICS ENGINE ---
total_trades = len(df)
wins = df[df['Realized_PL'] > 0]
losses = df[df['Realized_PL'] <= 0]

total_pl = df['Realized_PL'].sum()
total_risk_saved = df['Risk_Saved'].sum()
win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
profit_factor = (wins['Realized_PL'].sum() / abs(losses['Realized_PL'].sum())) if not losses.empty and losses['Realized_PL'].sum() != 0 else float('inf')

# --- EFFICIENCY RATIO ENGINE ---
fast_trades = df[df['Duration'] < 14]
slow_trades = df[df['Duration'] >= 14]

fast_pl_daily = (fast_trades['Realized_PL'] / fast_trades['Duration']).mean() if not fast_trades.empty else 0
slow_pl_daily = (slow_trades['Realized_PL'] / slow_trades['Duration']).mean() if not slow_trades.empty else 0

eff_insight = "Data insufficient for efficiency analysis."
eff_color = "gray"

if not fast_trades.empty and not slow_trades.empty:
    if fast_pl_daily > slow_pl_daily:
        ratio = fast_pl_daily / slow_pl_daily if slow_pl_daily > 0 else 1.0
        eff_insight = f"⚡️ **Velocity Alert:** Fast trades earn **{ratio:.1f}x** more per day than held trades. **Target 60% profit.**"
        eff_color = SUCCESS_COLOR
    else:
        eff_insight = "🐢 **Patience Pays:** Longer duration trades are currently yielding higher daily returns."
        eff_color = "#FFA726"

# --- METRICS ROW ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net P&L", f"${total_pl:,.0f}", delta_color="normal")
m2.metric("Win Rate", f"{win_rate:.0f}%", f"{len(wins)}W - {len(losses)}L")
m3.metric("Profit Factor", f"{profit_factor:.2f}")
m4.metric("Risk Averted", f"${total_risk_saved:,.0f}", delta_color="normal", help="Total capital saved by closing losers before Max Loss.")
m5.metric("Avg Duration", f"{df['Duration'].mean():.0f} Days")

st.markdown(f"<div style='background-color: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 5px; border-left: 3px solid {eff_color}; font-size: 14px; margin-bottom: 20px;'>{eff_insight}</div>", unsafe_allow_html=True)

# --- CHARTS (Matplotlib) ---
c1, c2 = st.columns(2)

df_sorted = df.sort_values(by="Exit_Date")

# 1. EQUITY CURVE
with c1:
    st.markdown("**💰 Equity Curve**")
    if not df.empty:
        df_sorted['Cumulative_PL'] = df_sorted['Realized_PL'].cumsum()
        
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.plot(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR, linewidth=2)
        ax.fill_between(df_sorted['Exit_Date'], df_sorted['Cumulative_PL'], color=SUCCESS_COLOR, alpha=0.1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    else:
        st.caption("No data.")

# 2. RISK AVERTED CURVE
with c2:
    st.markdown(f"**🛡️ Cumulative Risk Averted** <span style='font-size:0.8em; color:{SAVED_COLOR}'> (The 'Smart Exit' Chart)</span>", unsafe_allow_html=True)
    if not df.empty:
        df_sorted['Cumulative_Saved'] = df_sorted['Risk_Saved'].cumsum()
        
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        
        ax2.plot(df_sorted['Exit_Date'], df_sorted['Cumulative_Saved'], color=SAVED_COLOR, linewidth=2, linestyle='--')
        ax2.fill_between(df_sorted['Exit_Date'], df_sorted['Cumulative_Saved'], color=SAVED_COLOR, alpha=0.1)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
    else:
        st.caption("No losses managed yet.")

st.markdown("---")

# --- THE GALLERY (Trade Cards) ---
st.subheader("Trade History")

# Reverse order to show newest first
for i, row in df.iloc[::-1].iterrows():
    
    pl = row['Realized_PL']
    is_win = pl > 0
    is_scratch = (pl == 0)
    
    # Compact Colors
    if is_win:
        border_color = SUCCESS_COLOR
        card_bg = "rgba(0, 200, 83, 0.05)"
        pl_display = f"<span style='color:{SUCCESS_COLOR}'>+${pl:,.2f}</span>"
        risk_badge = ""
    elif is_scratch:
        border_color = NEUTRAL_COLOR
        card_bg = "rgba(117, 117, 117, 0.05)"
        pl_display = f"<span style='color:{NEUTRAL_COLOR}'>${pl:,.2f} (Scratch)</span>"
        risk_badge = ""
    else:
        border_color = WARNING_COLOR
        card_bg = "rgba(211, 47, 47, 0.05)"
        max_loss_val = row['Max_Loss_Trade']
        saved_val = row['Risk_Saved']
        pl_display = f"<span style='color:{WARNING_COLOR}'>-${abs(pl):,.2f}</span> <span style='font-size:0.8em; color:#666;'>/ -${max_loss_val:,.0f} Max</span>"
        risk_badge = f"""<div class="risk-badge">🛡️ Saved <strong>${saved_val:,.2f}</strong> by stopping out.</div>"""

    # Earnings display logic
    earnings_display = ""
    if pd.notnull(row['Earnings_Date']):
        e_date = row['Earnings_Date']
        # Check if Earnings fell INSIDE the trade window
        held_through = (e_date >= row['Entry_Date']) and (e_date <= row['Exit_Date'])
        
        if held_through:
            earnings_display = f"""<span class="earnings-badge earnings-danger">⚠️ HELD THRU EARNINGS ({e_date.strftime('%b %d')})</span>"""
        else:
            earnings_display = f"""<span style="font-size:10px; color:#666; margin-left:10px;">✅ Earnings: {e_date.strftime('%b %d')} (Avoided)</span>"""

    # Fix display for 0 strikes
    short_strike_display = f"{row['Short_Strike']:.0f}" if row['Short_Strike'] > 0 else "N/A"

    # --- HTML RENDER FIX ---
    # Using textwrap.dedent removes the indentation so Markdown renders HTML, not Code block.
    card_html = textwrap.dedent(f"""
        <div class="trade-card" style="border-left: 4px solid {border_color}; background-color: {card_bg}; padding: 10px; border-radius: 4px; margin-bottom: 8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; font-size: 16px; color:white;">
                    {row['Ticker']} 
                    <span style="font-size:0.7em; color:#888;">{row['Exit_Date'].strftime('%b %d')}</span>
                    {earnings_display}
                </h3>
                <h3 style="margin:0; font-size: 16px;">{pl_display}</h3>
            </div>
            {risk_badge}
            <div class="trade-details" style="display:flex; gap: 15px; margin-top: 5px; font-size: 12px; color: #aaa;">
                <span><strong>Strikes:</strong> {short_strike_display}/{row['Long_Strike']:.0f}</span>
                <span><strong>Duration:</strong> {int(row['Duration'])}d</span>
                <span><strong>Efficiency:</strong> ${row['Realized_PL']/row['Duration']:.2f}/day</span>
            </div>
        </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)
    
    with st.expander(f"Details", expanded=False):
        c_note, c_act = st.columns([4, 1])
        with c_note:
            st.write(f"**Notes:** {row.get('Notes', '-')}")
        with c_act:
            if st.button("Delete", key=f"del_{i}"):
                if delete_log_entry(drive_service, i):
                    st.rerun()
