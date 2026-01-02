import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import scipy.stats as si
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Options Lab")

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'
GRID_COLOR = '#444444'

# --- MATPLOTLIB STYLE ---
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3
})

# --- BLACK-SCHOLES SOLVER ---
def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates theoretical Put Price and Delta.
    S: Spot Price
    K: Strike Price
    T: Time to Expiry (years)
    r: Risk Free Rate
    sigma: Volatility (annualized)
    """
    try:
        if T <= 0 or sigma <= 0: 
            # Intrinsic value at expiry
            return max(K - S, 0), -1.0 if K > S else 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = (K * np.exp(-r * T) * si.norm.cdf(-d2)) - (S * si.norm.cdf(-d1))
        put_delta = si.norm.cdf(d1) - 1
        
        return put_price, put_delta
    except:
        return 0.0, 0.0

# --- SIMULATION ENGINE ---
@st.cache_data(ttl=3600)
def run_simulation(ticker, entry_delta, exit_delta_trigger, exit_spread_pct, exit_dte_trigger):
    # 1. Get Stock Data (2 Years)
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    if df.empty: return pd.DataFrame()
    
    # Calculate Volatility (30-day Rolling HV)
    df['Returns'] = df['Close'].pct_change()
    df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
    df = df.dropna()

    trades = []
    
    # Identify Entry Dates (Every Monday)
    df['DayOfWeek'] = df.index.dayofweek
    entry_dates = df[df['DayOfWeek'] == 0].index
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, entry_date in enumerate(entry_dates):
        if i % 5 == 0:
            progress_bar.progress((i + 1) / len(entry_dates))
            status_text.caption(f"Simulating Trade {i+1}/{len(entry_dates)}...")

        # --- SETUP TRADE ---
        entry_row = df.loc[entry_date]
        S_entry = entry_row['Close']
        sigma_entry = entry_row['HV']
        
        # Target Expiry ~45 DTE
        target_expiry = entry_date + timedelta(days=45)
        
        # Find closest trading day to expiry
        future_dates = df.index[df.index >= target_expiry]
        if future_dates.empty: continue # Not enough data left
        expiry_date = future_dates[0]
        
        T_entry = (expiry_date - entry_date).days / 365.0
        r = 0.045 # Assume 4.5% Risk Free Rate
        
        # FIND STRIKE based on Delta
        # Iterative Search for precise strike
        best_strike = S_entry
        min_diff = 1.0
        
        # Search range: 20% below to ATM
        for k in np.linspace(S_entry * 0.8, S_entry, 50):
            _, delta = black_scholes_put(S_entry, k, T_entry, r, sigma_entry)
            diff = abs(abs(delta) - abs(entry_delta))
            if diff < min_diff:
                min_diff = diff
                best_strike = k

        # Round strike
        strike_step = 5 if S_entry > 200 else 1
        short_strike = round(best_strike / strike_step) * strike_step
        long_strike = short_strike - 5.0 # Fixed width
        
        # Calculate Credit Received
        p_short, d_short = black_scholes_put(S_entry, short_strike, T_entry, r, sigma_entry)
        p_long, d_long = black_scholes_put(S_entry, long_strike, T_entry, r, sigma_entry)
        
        entry_credit = p_short - p_long
        
        if entry_credit < 0.10: continue # Skip invalid trades
        
        # --- MANAGE TRADE (Walk Forward) ---
        trade_res = {
            'entry_date': entry_date,
            'short_strike': short_strike,
            'credit': entry_credit,
            'exit_reason': 'Held to Expiry',
            'pnl': entry_credit,
            'exit_date': expiry_date
        }
        
        # Look at every day between Entry and Expiry
        trade_path = df.loc[entry_date:expiry_date]
        
        for curr_date, row in trade_path.iloc[1:].iterrows():
            S_curr = row['Close']
            T_curr = (expiry_date - curr_date).days / 365.0
            days_left = (expiry_date - curr_date).days
            
            # Use current HV for pricing (simplification of sticky delta)
            sigma_curr = row['HV']
            
            # Calc Current Value
            curr_p_short, curr_d_short = black_scholes_put(S_curr, short_strike, T_curr, r, sigma_curr)
            curr_p_long, _ = black_scholes_put(S_curr, long_strike, T_curr, r, sigma_curr)
            
            spread_val = curr_p_short - curr_p_long
            spread_pct = (spread_val / entry_credit) * 100
            
            # EXIT CHECKS
            
            # 1. DTE
            if days_left < exit_dte_trigger:
                trade_res['exit_reason'] = f"DTE < {exit_dte_trigger}"
                trade_res['pnl'] = entry_credit - spread_val
                trade_res['exit_date'] = curr_date
                break
                
            # 2. Delta
            if abs(curr_d_short) > exit_delta_trigger:
                trade_res['exit_reason'] = f"Delta > {exit_delta_trigger}"
                trade_res['pnl'] = entry_credit - spread_val
                trade_res['exit_date'] = curr_date
                break
                
            # 3. Spread Value (Stop Loss)
            if spread_pct > exit_spread_pct:
                trade_res['exit_reason'] = f"Max Loss ({exit_spread_pct}%)"
                trade_res['pnl'] = entry_credit - spread_val
                trade_res['exit_date'] = curr_date
                break
                
            # 4. Take Profit (50%)
            if spread_pct < 50:
                trade_res['exit_reason'] = "Profit Target (50%)"
                trade_res['pnl'] = entry_credit - spread_val
                trade_res['exit_date'] = curr_date
                break
                
        trades.append(trade_res)
        
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(trades)

# --- HEADER UI ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except:
        st.write("**DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Options Lab</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Backtesting & Optimization Engine (Simulation Mode)</p>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.write("") 

st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- CONTROLS ---
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    
    with c1:
        # UPDATED TICKER LIST
        ticker = st.selectbox("Select Ticker", ["SPY", "QQQ", "IWM", "GLD", "NVDA", "AAPL", "AMD", "TSLA", "MSFT", "AMZN"], index=0)
        st.caption("Data: Real Price History + Black-Scholes Sim")
        
    with c2:
        # Default Entry Delta is 0.20
        exit_delta = st.number_input("Exit Short Delta >", min_value=0.10, max_value=1.00, value=0.40, step=0.05, help="Close trade if short leg delta breaches this level.")
    
    with c3:
        exit_spread_pct = st.number_input("Exit Spread Value % >", min_value=100, max_value=500, value=200, step=10, help="Close if spread value hits this % of credit received. (200% = 1x Loss)")
        
    with c4:
        exit_dte = st.number_input("Exit DTE <", min_value=0, max_value=30, value=14, step=1, help="Close trade if fewer than X days remain.")
        
    run_btn = st.button("🔬 Run Simulation", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    with st.spinner(f"Simulating options chain history for {ticker}..."):
        # Fixed Entry Delta 0.20 (Standard for this strategy)
        results = run_simulation(ticker, 0.20, exit_delta, exit_spread_pct, exit_dte)
    
    if results.empty:
        st.error("Simulation failed. Ticker data might be unavailable.")
    else:
        # --- METRICS ---
        total_trades = len(results)
        wins = results[results['pnl'] > 0]
        losses = results[results['pnl'] <= 0]
        
        win_rate = len(wins) / total_trades * 100
        total_pnl = results['pnl'].sum() * 100 # Multiplier
        avg_win = wins['pnl'].mean() * 100 if not wins.empty else 0
        avg_loss = losses['pnl'].mean() * 100 if not losses.empty else 0
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P&L (1 Lot)", f"${total_pnl:,.2f}", delta_color="normal")
        m2.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W - {len(losses)}L")
        m3.metric("Avg Win", f"${avg_win:.2f}")
        m4.metric("Avg Loss", f"${avg_loss:.2f}")
        
        st.markdown("---")
        
        # --- CHARTS ---
        col_chart, col_dist = st.columns([2, 1])
        
        with col_chart:
            st.subheader("Equity Curve")
            results = results.sort_values("entry_date")
            results['cum_pnl'] = results['pnl'].cumsum() * 100
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(results['entry_date'], results['cum_pnl'], color=SUCCESS_COLOR, linewidth=2)
            ax.fill_between(results['entry_date'], results['cum_pnl'], color=SUCCESS_COLOR, alpha=0.1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.set_ylabel("Profit ($)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            
        with col_dist:
            st.subheader("Exit Reasons")
            exit_counts = results['exit_reason'].value_counts()
            
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            # Dynamic colors
            colors = [SUCCESS_COLOR if 'Profit' in idx or 'Held' in idx else WARNING_COLOR for idx in exit_counts.index]
            
            ax2.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'color': TEXT_COLOR})
            centre_circle = plt.Circle((0,0),0.70,fc=BG_COLOR)
            fig2.gca().add_artist(centre_circle)
            st.pyplot(fig2, use_container_width=True)
        
        # --- TRADE LIST ---
        st.subheader("Trade Log")
        st.dataframe(
            results[['entry_date', 'exit_date', 'short_strike', 'credit', 'exit_reason', 'pnl']],
            use_container_width=True,
            height=300
        )

# --- FOOTER NOTES ---
st.markdown("---")
st.caption("**Simulation Logic:** 45 DTE Entry, Sell 20-Delta Put. Options prices reconstructed using Black-Scholes model on realized volatility. This ensures 'lab' functionality without requiring external historical option databases.")
