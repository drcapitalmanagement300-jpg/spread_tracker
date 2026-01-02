import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Options Lab")

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'
GRID_COLOR = '#444444'

# Matplotlib Styling
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

# --- CACHED DATA LOADER ---
@st.cache_data(ttl=3600 * 24)  # Cache for 24 hours
def load_historical_options(ticker):
    """
    Loads historical options data from Philipp Dubach's repository.
    Uses Parquet format for speed.
    """
    try:
        url = f"https://static.philippdubach.com/data/options/{ticker}/options.parquet"
        
        # We only need specific columns to keep memory usage low
        columns = [
            'quote_date', 'expire_date', 'strike', 'option_type', 
            'implied_volatility', 'delta', 'bid', 'ask'
        ]
        
        df = pd.read_parquet(url, columns=columns)
        
        # Filter: Puts only
        df = df[df['option_type'] == 'P']
        
        # Dates
        df['quote_date'] = pd.to_datetime(df['quote_date'])
        df['expire_date'] = pd.to_datetime(df['expire_date'])
        df['dte'] = (df['expire_date'] - df['quote_date']).dt.days
        
        # Calculate Mid Price for easier simulation
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        
        return df
    except Exception as e:
        return None

# --- BACKTEST ENGINE ---
def run_backtest(df, ticker, exit_delta, exit_spread_pct, exit_dte):
    trades = []
    
    # Simulate Weekly Entries (Every Monday)
    unique_dates = sorted(df['quote_date'].unique())
    entry_dates = [d for d in unique_dates if pd.Timestamp(d).dayofweek == 0]
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    total_steps = len(entry_dates)
    
    for i, entry_date in enumerate(entry_dates):
        # Update UI every 10 steps to keep it snappy
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total_steps)
            status_text.caption(f"Simulating trade {i+1}/{total_steps}...")

        # --- 1. FIND ENTRY (Replicating Spread Sniper Logic) ---
        # Filter: 30-50 DTE
        daily_chain = df[
            (df['quote_date'] == entry_date) & 
            (df['dte'] >= 30) & 
            (df['dte'] <= 50)
        ]
        
        if daily_chain.empty: continue
        
        # Target: ~20 Delta (Proxy for 0.75x Expected Move)
        # In the screener, 0.75x EM usually lands around 20-25 Delta.
        target_delta = -0.25
        daily_chain = daily_chain.copy() # Avoid SettingWithCopy
        daily_chain['delta_diff'] = abs(abs(daily_chain['delta']) - abs(target_delta))
        
        # Get Short Leg
        short_leg = daily_chain.loc[daily_chain['delta_diff'].idxmin()]
        short_strike = short_leg['strike']
        expiry = short_leg['expire_date']
        
        # Get Long Leg ($5 wide)
        long_strike = short_strike - 5.0
        long_candidates = daily_chain[abs(daily_chain['strike'] - long_strike) < 0.5]
        
        if long_candidates.empty: continue
        long_leg = long_candidates.iloc[0]
        
        # Calculate Credit
        entry_credit = short_leg['mid_price'] - long_leg['mid_price']
        
        # Filter: Min Credit $0.70 (Screener Rule)
        if entry_credit < 0.70: continue
        
        # --- 2. MANAGE TRADE (Step forward in time) ---
        trade_result = {
            'entry_date': entry_date,
            'short_strike': short_strike,
            'credit': entry_credit,
            'exit_reason': 'Held to Expiry',
            'pnl': entry_credit, # Default to max profit
            'exit_date': expiry
        }
        
        # Look at future dates for this specific spread
        future_data = df[
            (df['expire_date'] == expiry) & 
            (df['quote_date'] > entry_date) & 
            (df['quote_date'] <= expiry)
        ]
        
        # Group by date to see the spread value each day
        for date, group in future_data.groupby('quote_date'):
            # Reconstruct the spread
            curr_short = group[group['strike'] == short_strike]
            curr_long = group[group['strike'] == long_strike] # Approximate
            
            if curr_short.empty: continue
            
            # Get metrics
            curr_short_row = curr_short.iloc[0]
            curr_delta = abs(curr_short_row['delta'])
            curr_dte = (expiry - date).days
            
            # Approximate spread value
            s_price = curr_short_row['mid_price']
            l_price = 0.0
            if not curr_long.empty:
                l_price = curr_long.iloc[0]['mid_price']
            
            spread_value = s_price - l_price
            spread_pct = (spread_value / entry_credit) * 100
            
            # --- 3. CHECK EXIT CRITERIA ---
            
            # A. DTE Exit
            if curr_dte < exit_dte:
                trade_result['exit_reason'] = f"DTE < {exit_dte}"
                trade_result['pnl'] = entry_credit - spread_value
                trade_result['exit_date'] = date
                break
                
            # B. Delta Breach
            if curr_delta > exit_delta:
                trade_result['exit_reason'] = f"Delta > {exit_delta:.2f}"
                trade_result['pnl'] = entry_credit - spread_value
                trade_result['exit_date'] = date
                break
                
            # C. Stop Loss / Spread Value Explosion
            if spread_pct > exit_spread_pct:
                trade_result['exit_reason'] = f"Value > {exit_spread_pct}%"
                trade_result['pnl'] = entry_credit - spread_value
                trade_result['exit_date'] = date
                break
                
            # D. Profit Take (Standard 50%)
            if spread_pct < 50: # If value drops below 50% of credit received
                trade_result['exit_reason'] = "Profit Target (50%)"
                trade_result['pnl'] = entry_credit - spread_value
                trade_result['exit_date'] = date
                break
        
        trades.append(trade_result)

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
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Backtesting & Optimization Engine</p>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.write("") 

st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- CONTROLS ---
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    
    with c1:
        ticker = st.selectbox("Select Ticker", ["SPY", "QQQ", "IWM", "GLD", "NVDA", "AAPL", "AMD", "TSLA", "MSFT", "AMZN"], index=0)
        st.caption("Data Source: Philipp Dubach Repo")
        
    with c2:
        exit_delta = st.number_input("Exit Short Delta >", min_value=0.10, max_value=1.00, value=0.40, step=0.05, help="Close trade if short leg delta breaches this level.")
    
    with c3:
        exit_spread_pct = st.number_input("Exit Spread Value % >", min_value=100, max_value=500, value=200, step=10, help="Close if spread value hits this % of credit received. (200% = 1x Loss)")
        
    with c4:
        exit_dte = st.number_input("Exit DTE <", min_value=0, max_value=30, value=14, step=1, help="Close trade if fewer than X days remain.")
        
    run_btn = st.button("🔬 Run Simulation", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    with st.spinner(f"Downloading historical chain for {ticker}... (This may take a moment)"):
        df_hist = load_historical_options(ticker)
    
    if df_hist is None or df_hist.empty:
        st.error(f"Could not load data for {ticker}. Please try SPY or QQQ.")
    else:
        st.success(f"Data Loaded! Running simulation on {len(df_hist):,} rows...")
        
        # Run Sim
        results = run_backtest(df_hist, ticker, exit_delta, exit_spread_pct, exit_dte)
        
        if results.empty:
            st.warning("No valid trades found with current Screener settings in this period.")
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
st.caption("**Note:** Simulation assumes standard 'Spread Sniper' entry rules (30-50 DTE, ~$0.70 Credit). Mid-price used for fills. Data provided by Philipp Dubach options repo.")
