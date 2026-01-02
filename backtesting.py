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
    """
    try:
        S, K, T, sigma = float(S), float(K), float(T), float(sigma)
        if T <= 0 or sigma <= 0: 
            return max(K - S, 0.0), -1.0 if K > S else 0.0
        
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
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="10y") 
        if df.empty: return pd.DataFrame()
        
        # 1. FIX TIMEZONES (Crucial for YFinance)
        df.index = df.index.tz_localize(None)
        
        # 2. Volatility
        df['Returns'] = df['Close'].pct_change()
        df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
    except Exception as e:
        return pd.DataFrame()

    trades = []
    entry_dates = df.index
    
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_days = len(entry_dates)
    
    r = 0.045 # Risk Free Rate Assumption
    
    for i, entry_date in enumerate(entry_dates):
        if i % 50 == 0:
            progress_bar.progress((i + 1) / total_days)
            status_text.caption(f"Simulating Trade {i+1}/{total_days}...")

        try:
            entry_row = df.loc[entry_date]
            S_entry = float(entry_row['Close'])
            sigma_entry = float(entry_row['HV'])
            
            # Target Expiry (45 Days)
            target_expiry = entry_date + timedelta(days=45)
            future_data = df.loc[entry_date:]
            valid_expiries = future_data.index[future_data.index >= target_expiry]
            
            if valid_expiries.empty: break
            expiry_date = valid_expiries[0]
            T_entry = (expiry_date - entry_date).days / 365.0
            
            # --- ROBUST STRIKE FINDER (Iterative) ---
            # We scan from 20% below spot up to Spot price to find the 20 Delta strike
            # This is 100% reliable compared to the approximation formula
            best_strike = S_entry * 0.8
            min_delta_diff = 1.0
            
            # Scan 40 potential strikes
            scan_strikes = np.linspace(S_entry * 0.70, S_entry, 40)
            for k_candidate in scan_strikes:
                _, d_test = black_scholes_put(S_entry, k_candidate, T_entry, r, sigma_entry)
                diff = abs(abs(d_test) - abs(entry_delta))
                if diff < min_delta_diff:
                    min_delta_diff = diff
                    best_strike = k_candidate

            # Snap to grid ($5 steps for >$200, else $1)
            strike_step = 5.0 if S_entry > 200 else 1.0
            short_strike = round(best_strike / strike_step) * strike_step
            long_strike = short_strike - 5.0
            
            # Calculate Entry Credit
            p_short, d_short = black_scholes_put(S_entry, short_strike, T_entry, r, sigma_entry)
            p_long, _ = black_scholes_put(S_entry, long_strike, T_entry, r, sigma_entry)
            entry_credit = p_short - p_long
            
            # Filter: Minimum Credit $0.10
            if entry_credit < 0.10: continue 
            
            # --- TRADE EXECUTION ---
            trade_res = {
                'entry_date': entry_date,
                'short_strike': short_strike,
                'credit': entry_credit,
                'exit_reason': 'Held to Expiry',
                'pnl': entry_credit,
                'exit_date': expiry_date
            }
            
            # Walk Forward
            trade_path = df.loc[entry_date:expiry_date].iloc[1:]
            
            for curr_date, row in trade_path.iterrows():
                S_curr = float(row['Close'])
                T_curr = (expiry_date - curr_date).days / 365.0
                days_left = (expiry_date - curr_date).days
                sigma_curr = float(row['HV'])
                
                curr_p_short, curr_d_short = black_scholes_put(S_curr, short_strike, T_curr, r, sigma_curr)
                curr_p_long, _ = black_scholes_put(S_curr, long_strike, T_curr, r, sigma_curr)
                
                spread_val = curr_p_short - curr_p_long
                spread_pct = (spread_val / entry_credit) * 100
                
                # Exits
                if days_left < exit_dte_trigger:
                    trade_res.update({'exit_reason': f"DTE < {exit_dte_trigger}", 'pnl': entry_credit - spread_val, 'exit_date': curr_date})
                    break
                if abs(curr_d_short) > exit_delta_trigger:
                    trade_res.update({'exit_reason': f"Delta > {exit_delta_trigger}", 'pnl': entry_credit - spread_val, 'exit_date': curr_date})
                    break
                if spread_pct > exit_spread_pct:
                    trade_res.update({'exit_reason': f"Max Loss ({exit_spread_pct}%)", 'pnl': entry_credit - spread_val, 'exit_date': curr_date})
                    break
                if spread_pct < 50:
                    trade_res.update({'exit_reason': "Profit Target (50%)", 'pnl': entry_credit - spread_val, 'exit_date': curr_date})
                    break
                    
            trades.append(trade_res)
        except: continue
        
    progress_bar.empty()
    status_text.empty()
    
    # Clean up results for display
    results = pd.DataFrame(trades)
    if not results.empty:
        # Date Formatting
        results['entry_date'] = pd.to_datetime(results['entry_date']).dt.strftime('%Y-%m-%d')
        results['exit_date'] = pd.to_datetime(results['exit_date']).dt.strftime('%Y-%m-%d')
    return results

# --- HEADER UI ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""<div style='text-align: left; padding-top: 10px;'><h1 style='margin-bottom: 0px;'>Options Lab</h1><p style='color: gray;'>Large Scale Backtesting (10yr Simulation)</p></div>""", unsafe_allow_html=True)
with header_col3: st.write("") 
st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- CONTROLS ---
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1: ticker = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "GLD", "NVDA", "AAPL", "AMD", "TSLA", "MSFT", "AMZN"], index=0)
    with c2: exit_delta = st.number_input("Exit Short Delta >", 0.1, 1.0, 0.60, 0.05)
    with c3: exit_spread_pct = st.number_input("Exit Spread Value % >", 100, 1000, 300, 50)
    with c4: exit_dte = st.number_input("Exit DTE <", 0, 30, 7)
    run_btn = st.button("🔬 Run 10-Year Simulation", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    with st.spinner(f"Simulating daily trades for {ticker} (10 Years)..."):
        results = run_simulation(ticker, 0.20, exit_delta, exit_spread_pct, exit_dte)
    
    if results.empty:
        st.error("Simulation failed. No trades found. Try a different ticker or check connection.")
    else:
        # Metrics
        total_trades = len(results)
        wins = results[results['pnl'] > 0]
        losses = results[results['pnl'] <= 0]
        win_rate = len(wins) / total_trades * 100
        total_pnl = results['pnl'].sum() * 100
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P&L (1 Lot)", f"${total_pnl:,.2f}")
        m2.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W - {len(losses)}L")
        m3.metric("Avg Win", f"${wins['pnl'].mean()*100:.2f}")
        m4.metric("Avg Loss", f"${losses['pnl'].mean()*100:.2f}")
        st.markdown("---")
        
        # Charts
        col_chart, col_dist = st.columns([2, 1])
        with col_chart:
            st.subheader(f"Equity Curve ({total_trades} Trades)")
            results['plot_date'] = pd.to_datetime(results['entry_date'])
            results = results.sort_values("plot_date")
            results['cum_pnl'] = results['pnl'].cumsum() * 100
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(results['plot_date'], results['cum_pnl'], color=SUCCESS_COLOR, linewidth=1.5)
            ax.fill_between(results['plot_date'], results['cum_pnl'], color=SUCCESS_COLOR, alpha=0.1)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.set_ylabel("Profit ($)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            
        with col_dist:
            st.subheader("Exit Reasons")
            exit_counts = results['exit_reason'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            colors = [SUCCESS_COLOR if 'Profit' in idx or 'Held' in idx else WARNING_COLOR for idx in exit_counts.index]
            ax2.pie(exit_counts, labels=exit_counts.index, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'color': TEXT_COLOR})
            centre_circle = plt.Circle((0,0),0.70,fc=BG_COLOR)
            fig2.gca().add_artist(centre_circle)
            st.pyplot(fig2, use_container_width=True)
        
        # Trade Log
        st.subheader("Detailed Log")
        st.dataframe(
            results[['entry_date', 'exit_date', 'short_strike', 'credit', 'exit_reason', 'pnl']],
            use_container_width=True,
            height=300
        )
st.markdown("---")
st.caption("**Scale:** 10-Year History | **Frequency:** Daily Entry | **Model:** Black-Scholes Recreation")
