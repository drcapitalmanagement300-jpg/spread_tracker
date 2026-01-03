import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import scipy.stats as si
import time
import itertools
from datetime import timedelta
import concurrent.futures

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Options Lab")

# --- CONSTANTS ---
# Top 5 Highest Liquidity/IV Stocks (Non-Index)
BASKET_TICKERS = ["TSLA", "NVDA", "AAPL", "AMD", "AMZN"]

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

# --- MATH HELPERS ---
def black_scholes_put(S, K, T, r, sigma):
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

# --- DATA LOADING (CACHED) ---
@st.cache_data(ttl=3600*24)
def get_basket_data(tickers):
    """Fetches data for ALL basket tickers at once."""
    data_map = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="10y")
            if df.empty: continue
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df['Returns'] = df['Close'].pct_change()
            df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            data_map[t] = df
        except: continue
    return data_map

# --- CORE SIGNAL GENERATOR (Single Ticker) ---
def generate_signals_for_ticker(ticker, df, entry_delta, exit_dte, stop_loss_pct, profit_target_pct, scan_resolution_low=False):
    """
    Calculates potential trades assuming INFINITE capital. 
    The Portfolio Manager will later filter these based on actual cash.
    """
    signals = []
    entry_dates = df.index
    r = 0.045
    
    # Speed Optimization: Reduce strike scan steps for optimizer
    strike_steps = 10 if scan_resolution_low else 30
    
    for entry_date in entry_dates:
        try:
            # 1. Entry Logic
            row = df.loc[entry_date]
            S = float(row['Close'])
            sigma = float(row['HV'])
            
            target_expiry = entry_date + timedelta(days=45)
            future = df.loc[entry_date:]
            valid_exps = future.index[future.index >= target_expiry]
            if valid_exps.empty: continue
            
            expiry = valid_exps[0]
            T = (expiry - entry_date).days / 365.0
            
            # Find Strike
            best_strike = S * 0.8
            min_diff = 1.0
            candidates = np.linspace(S * 0.70, S, strike_steps)
            
            for k in candidates:
                _, d = black_scholes_put(S, k, T, r, sigma)
                if abs(abs(d) - abs(entry_delta)) < min_diff:
                    min_diff = abs(abs(d) - abs(entry_delta))
                    best_strike = k
            
            # Grid Snap
            step = 5.0 if S > 200 else 1.0
            short_k = round(best_strike / step) * step
            long_k = short_k - 5.0
            
            # Pricing
            p_s, d_s = black_scholes_put(S, short_k, T, r, sigma)
            p_l, _ = black_scholes_put(S, long_k, T, r, sigma)
            credit = p_s - p_l
            
            if credit < 0.10: continue # Filter junk
            
            # 2. Walk Forward (Path Dependency)
            trade_path = df.loc[entry_date:expiry].iloc[1:]
            
            exit_date = expiry
            exit_reason = "Expired"
            final_pnl = credit # Default: Full profit
            
            for curr_date, path_row in trade_path.iterrows():
                S_curr = float(path_row['Close'])
                days_left = (expiry - curr_date).days
                T_curr = days_left / 365.0
                sigma_curr = float(path_row['HV'])
                
                # Current Spread Value
                cp_s, _ = black_scholes_put(S_curr, short_k, T_curr, r, sigma_curr)
                cp_l, _ = black_scholes_put(S_curr, long_k, T_curr, r, sigma_curr)
                spread_val = cp_s - cp_l
                
                profit_pct = ((credit - spread_val) / credit) * 100
                loss_pct = (spread_val / credit) * 100
                
                # Check Exits
                hit = False
                if days_left <= exit_dte:
                    hit = True; exit_reason = f"DTE {exit_dte}"
                elif profit_pct >= profit_target_pct:
                    hit = True; exit_reason = "Profit Target"
                elif loss_pct >= stop_loss_pct:
                    hit = True; exit_reason = "Stop Loss"
                
                if hit:
                    exit_date = curr_date
                    final_pnl = credit - spread_val
                    break
            
            signals.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'gross_pnl': final_pnl,
                'exit_reason': exit_reason,
                'margin': 500.0, # Standard $5 wide spread
                'entry_delta': entry_delta
            })
            
        except: continue
    return signals

# --- PORTFOLIO MANAGER (The "Accountant") ---
def run_portfolio_simulation(data_map, entry_delta, exit_dte, stop_loss, profit_target, 
                           start_cap, monthly_add, slippage, fees, 
                           progress_bar_slot=None):
    
    # 1. GENERATE SIGNALS (Parallel)
    all_signals = []
    
    # Use ThreadPool to run 5 tickers at once
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ticker, df in data_map.items():
            # Pass low_res=True for speed during optimization
            futures.append(executor.submit(generate_signals_for_ticker, ticker, df, entry_delta, exit_dte, stop_loss, profit_target, True))
        
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())
            
    # Sort all signals by entry date to simulate timeline
    all_signals.sort(key=lambda x: x['entry_date'])
    
    # 2. RUN TIMELINE
    current_cash = start_cap
    active_trades = [] # List of {'exit_date', 'pnl', 'margin_locked'}
    completed_trades = []
    
    # Create timeline (Daily steps)
    if not all_signals: return pd.DataFrame(), 0.0
    
    start_date = all_signals[0]['entry_date']
    end_date = all_signals[-1]['exit_date']
    timeline = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Convert signals to a Queue for efficient popping
    signal_queue = all_signals
    signal_idx = 0
    total_signals = len(signal_queue)
    
    days_since_contrib = 0
    
    for i, today in enumerate(timeline):
        if progress_bar_slot and i % 50 == 0:
            progress_bar_slot.progress((i+1) / len(timeline))
            
        # A. Monthly Contribution
        days_since_contrib += 1
        if days_since_contrib >= 30:
            current_cash += monthly_add
            days_since_contrib = 0
            
        # B. Process Exits (Unlock Capital)
        # Filter active trades: Keep only those that haven't exited yet
        # We must identify trades that exit TODAY
        still_active = []
        for t in active_trades:
            if t['exit_date'] <= today:
                # Trade Closes
                # Apply Friction
                friction = (slippage * 4) + fees
                net_pnl = t['gross_pnl'] - (friction / 100.0) # Convert to share price
                realized_dollar_pnl = net_pnl * 100 # 1 contract = 100 shares
                
                # Return Margin + Profit to pool
                current_cash += (t['margin'] + realized_dollar_pnl)
                
                # Log
                completed_trades.append({
                    'entry_date': t['entry_date'],
                    'exit_date': t['exit_date'],
                    'ticker': t['ticker'],
                    'pnl_dollars': realized_dollar_pnl,
                    'reason': t['exit_reason'],
                    'balance': current_cash
                })
            else:
                still_active.append(t)
        active_trades = still_active
        
        # C. Process Entries (Lock Capital)
        # Check signals that happen TODAY
        while signal_idx < total_signals:
            sig = signal_queue[signal_idx]
            if sig['entry_date'] > today:
                break # Future trade, stop checking
            
            if sig['entry_date'] == today:
                # Try to take trade
                margin_req = sig['margin']
                if current_cash >= margin_req:
                    current_cash -= margin_req
                    active_trades.append(sig)
            
            signal_idx += 1
            
    return pd.DataFrame(completed_trades), current_cash

# --- UI HEADER ---
st.markdown("## Options Lab: Portfolio Optimizer")
st.markdown(f"**Universe:** {', '.join(BASKET_TICKERS)} | **Strategy:** Credit Spreads")
st.markdown("---")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Simulation Settings")
    start_cap = st.number_input("Starting Capital ($)", 1000, 1000000, 10000, 1000)
    monthly_add = st.number_input("Monthly Contribution ($)", 0, 10000, 500, 50)
    
    st.divider()
    st.caption("Friction Costs")
    slippage = st.number_input("Slippage ($/leg)", 0.00, 1.00, 0.01, 0.01)
    fees = st.number_input("Fees ($/trade)", 0.00, 10.00, 0.00, 0.10)
    
    st.divider()
    if st.button("Clear Cache"):
        st.cache_data.clear()

# --- TABS ---
tab_manual, tab_auto = st.tabs(["Manual Backtest", "Auto-Optimizer"])

# DATA FETCH (Global)
if 'data_map' not in st.session_state:
    with st.spinner("Initializing Market Data (Top 5 Tickers)..."):
        st.session_state.data_map = get_basket_data(BASKET_TICKERS)

# --- MANUAL LAB ---
with tab_manual:
    c1, c2, c3, c4 = st.columns(4)
    with c1: e_delta = st.number_input("Entry Delta", 0.1, 0.5, 0.20, 0.05)
    with c2: stop_loss = st.number_input("Stop Loss %", 100, 500, 300, 50)
    with c3: profit_target = st.number_input("Profit Target %", 10, 90, 50, 10)
    with c4: exit_dte = st.number_input("Exit DTE", 0, 30, 21)
    
    if st.button("Run Portfolio Simulation", type="primary"):
        prog_bar = st.progress(0, text="Simulating timeline...")
        
        df_res, final_bal = run_portfolio_simulation(
            st.session_state.data_map, e_delta, exit_dte, stop_loss, profit_target,
            start_cap, monthly_add, slippage, fees, prog_bar
        )
        prog_bar.empty()
        
        if not df_res.empty:
            tot_profit = df_res['pnl_dollars'].sum()
            win_rate = len(df_res[df_res['pnl_dollars']>0]) / len(df_res) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Balance", f"${final_bal:,.0f}", delta=f"${(final_bal - start_cap):,.0f}")
            m2.metric("Total Trades", len(df_res))
            m3.metric("Win Rate", f"{win_rate:.1f}%")
            
            st.line_chart(df_res, x='exit_date', y='balance', color=SUCCESS_COLOR)
            st.dataframe(df_res.sort_values('exit_date', ascending=False), use_container_width=True)
        else:
            st.warning("No trades generated.")

# --- OPTIMIZER ---
with tab_auto:
    st.info("This will find the best Entry/Exit combination for the entire 5-stock portfolio.")
    
    # OPTIMIZER STATE
    if 'opt_queue' not in st.session_state:
        st.session_state.opt_queue = []
    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = []
    
    if st.button("Start Optimization", type="primary"):
        # Grid Setup
        deltas = [0.15, 0.20, 0.30]
        dtes = [21, 7, 0]
        targets = [50, 25]
        stops = [200, 300]
        
        grid = list(itertools.product(deltas, dtes, targets, stops))
        st.session_state.opt_queue = grid
        st.session_state.opt_results = []
        st.session_state.is_running = True
        st.rerun()

    if st.button("Stop"):
        st.session_state.is_running = False
        st.session_state.opt_queue = []
        st.rerun()

    # WORKER
    if st.session_state.get('is_running', False):
        if not st.session_state.opt_queue:
            st.session_state.is_running = False
            st.rerun()
            
        # Pop Config
        cfg = st.session_state.opt_queue.pop(0)
        d_delta, d_dte, d_target, d_stop = cfg
        
        # UI Status
        remaining = len(st.session_state.opt_queue)
        total_runs = len(st.session_state.opt_results) + remaining + 1
        curr_run = len(st.session_state.opt_results) + 1
        
        st.write(f"**Processing Config {curr_run}/{total_runs}**")
        st.text(f"Delta: {d_delta} | DTE: {d_dte} | Target: {d_target}% | Stop: {d_stop}%")
        
        main_bar = st.progress(curr_run / total_runs)
        inner_bar = st.progress(0, text="Simulating Days...")
        
        # Run
        df_r, bal = run_portfolio_simulation(
            st.session_state.data_map, d_delta, d_dte, d_stop, d_target,
            start_cap, monthly_add, slippage, fees, inner_bar
        )
        
        # Save
        if not df_r.empty:
            roi = ((bal - start_cap) / start_cap) * 100
            wr = len(df_r[df_r['pnl_dollars']>0]) / len(df_r) * 100
            dd_series = (df_r['balance'] - df_r['balance'].cummax()) / df_r['balance'].cummax()
            max_dd = dd_series.min() * 100
            
            st.session_state.opt_results.append({
                "Delta": d_delta,
                "DTE": d_dte,
                "Target": d_target,
                "Stop": d_stop,
                "Final Balance": bal,
                "ROI %": roi,
                "Win Rate": wr,
                "Max DD %": max_dd
            })
            
        st.rerun()

    # RESULTS DISPLAY
    if st.session_state.opt_results and not st.session_state.get('is_running', False):
        res_df = pd.DataFrame(st.session_state.opt_results)
        res_df = res_df.sort_values("Final Balance", ascending=False)
        
        st.subheader("Top Performing Configurations")
        
        # Winner
        best = res_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Balance", f"${best['Final Balance']:,.0f}")
        c2.metric("Configuration", f"Delta {best['Delta']} | {int(best['DTE'])} DTE")
        c3.metric("Win Rate", f"{best['Win Rate']:.1f}%")
        
        st.dataframe(res_df.style.format({
            "Final Balance": "${:,.0f}",
            "ROI %": "{:.1f}%",
            "Win Rate": "{:.1f}%",
            "Max DD %": "{:.1f}%"
        }).background_gradient(subset="Final Balance", cmap="Greens"), use_container_width=True)
