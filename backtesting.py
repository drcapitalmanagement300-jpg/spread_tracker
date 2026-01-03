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

# --- MATH HELPERS (VECTORIZED) ---
def vectorized_black_scholes(S, K, T, r, sigma, type='put'):
    """
    Performs Black-Scholes calculation on entire Arrays at once.
    Massively faster than looping.
    """
    # Ensure inputs are numpy arrays for vector math
    S = np.array(S, dtype=float)
    K = np.array(K, dtype=float)
    T = np.array(T, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    # Avoid div by zero / log of zero errors
    # We use a mask to handle invalid math safely without crashing
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    # Cumulative Distribution Function
    # scipy.stats.norm.cdf works on arrays automatically
    nd1 = si.norm.cdf(-d1)
    nd2 = si.norm.cdf(-d2)
    
    price = (K * np.exp(-r * T) * nd2) - (S * nd1)
    delta = nd1 - 1
    
    # Clean up invalid results (expired options or zero vol)
    # If T <= 0, Price is Intrinsic Value (Max(K-S, 0))
    intrinsic_val = np.maximum(K - S, 0.0)
    price = np.where(T <= 0, intrinsic_val, price)
    
    # Final NaN catch
    price = np.nan_to_num(price)
    delta = np.nan_to_num(delta)
    
    return price, delta

# --- DATA LOADING ---
@st.cache_data(ttl=3600*24)
def get_basket_data(tickers):
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

# --- CORE SIGNAL GENERATOR (OPTIMIZED) ---
def generate_signals_for_ticker(ticker, df, entry_delta, exit_dte, stop_loss_pct, profit_target_pct, scan_resolution_low=False):
    """
    Optimized Signal Generator using Vectorized Operations.
    Removes inner loops for massive speed gains.
    """
    signals = []
    r = 0.045
    strike_steps = 10 if scan_resolution_low else 30
    
    # Pre-calculate common columns to avoid .loc overhead in loop
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    
    # Helper to map date to integer index for speed
    date_map = {d: i for i, d in enumerate(dates)}
    
    for i, entry_date in enumerate(dates):
        try:
            # 1. SETUP
            S = close_prices[i]
            sigma = hvs[i]
            
            # Fast Forward to Expiry (Approx 45 days)
            # Find index 45 days later (approx)
            target_date = entry_date + timedelta(days=45)
            
            # Fast lookup of next valid date
            # We can't do exact index math because of weekends/holidays, 
            # so we search the index efficiently
            future_slice = df.index[i:]
            valid_exps = future_slice[future_slice >= target_date]
            
            if valid_exps.empty: continue
            expiry = valid_exps[0]
            
            # Get integer index of expiry for slicing
            expiry_idx = date_map[expiry]
            
            T = (expiry - entry_date).days / 365.0
            
            # 2. STRIKE SELECTION (Vectorized)
            # Create array of 30 candidates
            candidates = np.linspace(S * 0.70, S, strike_steps)
            
            # One BS call for all 30 strikes
            _, d_arr = vectorized_black_scholes(
                np.full(strike_steps, S), 
                candidates, 
                np.full(strike_steps, T), 
                r, 
                np.full(strike_steps, sigma)
            )
            
            # Find closest delta
            best_idx = np.argmin(np.abs(np.abs(d_arr) - abs(entry_delta)))
            best_strike = candidates[best_idx]
            
            # Snap to grid
            step = 5.0 if S > 200 else 1.0
            short_k = round(best_strike / step) * step
            long_k = short_k - 5.0
            
            # 3. PRICING (Scalar is fine here, it's just one trade)
            p_s, _ = vectorized_black_scholes(S, short_k, T, r, sigma)
            p_l, _ = vectorized_black_scholes(S, long_k, T, r, sigma)
            credit = float(p_s - p_l)
            
            if credit < 0.10: continue 

            # 4. WALK FORWARD (Vectorized)
            # Instead of looping day by day, we slice the arrays
            # Path from Entry+1 to Expiry
            path_S = close_prices[i+1 : expiry_idx+1]
            path_sigma = hvs[i+1 : expiry_idx+1]
            path_dates = dates[i+1 : expiry_idx+1]
            
            if len(path_S) == 0: continue
            
            # Calculate Time to Expiry for every day in the path at once
            # (Expiry - CurrentDate).days
            path_days_left = np.array([(expiry - d).days for d in path_dates])
            path_T = path_days_left / 365.0
            
            # Calculate Option Prices for the WHOLE path at once
            path_ps, _ = vectorized_black_scholes(path_S, short_k, path_T, r, path_sigma)
            path_pl, _ = vectorized_black_scholes(path_S, long_k, path_T, r, path_sigma)
            path_spreads = path_ps - path_pl
            
            # Calculate Metrics Arrays
            profit_pcts = ((credit - path_spreads) / credit) * 100
            loss_pcts = (path_spreads / credit) * 100
            
            # Logic Masks (Boolean Arrays)
            mask_dte = path_days_left <= exit_dte
            mask_profit = profit_pcts >= profit_target_pct
            mask_stop = loss_pcts >= stop_loss_pct
            
            # Combine triggers
            # We want the FIRST day where ANY condition is met
            triggers = mask_dte | mask_profit | mask_stop
            
            exit_date = expiry
            exit_reason = "Expired"
            final_pnl = credit
            
            if triggers.any():
                # Get index of first True
                first_hit_idx = np.argmax(triggers)
                
                exit_date = path_dates[first_hit_idx]
                final_pnl = credit - path_spreads[first_hit_idx]
                
                # Determine Reason
                if mask_stop[first_hit_idx]: exit_reason = "Stop Loss"
                elif mask_profit[first_hit_idx]: exit_reason = "Profit Target"
                elif mask_dte[first_hit_idx]: exit_reason = f"DTE {exit_dte}"

            signals.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'gross_pnl': float(final_pnl),
                'exit_reason': exit_reason,
                'margin': 500.0,
                'entry_delta': entry_delta
            })
            
        except Exception: continue
        
    return signals

# --- PORTFOLIO MANAGER ---
def run_portfolio_simulation(data_map, entry_delta, exit_dte, stop_loss, profit_target, 
                           start_cap, monthly_add, slippage, fees, 
                           progress_bar_slot=None):
    
    # 1. GENERATE SIGNALS (Parallel & Vectorized)
    all_signals = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ticker, df in data_map.items():
            futures.append(executor.submit(generate_signals_for_ticker, ticker, df, entry_delta, exit_dte, stop_loss, profit_target, True))
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())
            
    all_signals.sort(key=lambda x: x['entry_date'])
    
    current_cash = start_cap
    active_trades = []
    completed_trades = []
    
    if not all_signals: return pd.DataFrame(), 0.0
    
    timeline = pd.date_range(start=all_signals[0]['entry_date'], end=all_signals[-1]['exit_date'], freq='D')
    signal_queue = all_signals
    signal_idx = 0
    total_signals = len(signal_queue)
    days_since_contrib = 0
    total_days = len(timeline)

    # 2. RUN TIMELINE
    for i, today in enumerate(timeline):
        
        if progress_bar_slot and i % (max(1, int(total_days * 0.05))) == 0:
            progress_bar_slot.progress((i + 1) / total_days, text=f"Simulating Day {i+1}/{total_days}")
            
        days_since_contrib += 1
        if days_since_contrib >= 30:
            current_cash += monthly_add
            days_since_contrib = 0
            
        still_active = []
        for t in active_trades:
            if t['exit_date'] <= today:
                friction = (slippage * 4) + fees
                net_pnl = t['gross_pnl'] - (friction / 100.0)
                realized_dollar_pnl = net_pnl * 100
                current_cash += (t['margin'] + realized_dollar_pnl)
                
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
        
        while signal_idx < total_signals:
            sig = signal_queue[signal_idx]
            if sig['entry_date'] > today: break
            if sig['entry_date'] == today:
                if current_cash >= sig['margin']:
                    current_cash -= sig['margin']
                    active_trades.append(sig)
            signal_idx += 1
            
    return pd.DataFrame(completed_trades), current_cash

# --- HEADER UI ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""<div style='text-align: left; padding-top: 10px;'><h1 style='margin-bottom: 0px;'>Options Lab</h1><p style='color: gray;'>Portfolio Simulation & Optimizer</p></div>""", unsafe_allow_html=True)
with header_col3: st.write("") 
st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
st.markdown(f"**Universe:** {', '.join(BASKET_TICKERS)} | **Strategy:** Credit Spreads")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Simulation Settings")
    start_cap = st.number_input("Starting Capital ($)", 1000, 1000000, 10000, 1000)
    monthly_add = st.number_input("Monthly Contribution ($)", 0, 10000, 500, 50)
    st.divider()
    slippage = st.number_input("Slippage ($/leg)", 0.00, 1.00, 0.01, 0.01)
    fees = st.number_input("Fees ($/trade)", 0.00, 10.00, 0.00, 0.10)
    st.divider()
    if st.button("Clear Cache"): st.cache_data.clear()

tab_manual, tab_auto = st.tabs(["Manual Backtest", "Auto-Optimizer"])

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
        prog_bar = st.progress(0, text="Initializing...")
        
        df_res, final_bal = run_portfolio_simulation(
            st.session_state.data_map, e_delta, exit_dte, stop_loss, profit_target,
            start_cap, monthly_add, slippage, fees, prog_bar
        )
        prog_bar.progress(1.0, text="Done!")
        time.sleep(0.5)
        prog_bar.empty()
        
        if not df_res.empty:
            win_rate = len(df_res[df_res['pnl_dollars']>0]) / len(df_res) * 100
            m1, m2, m3 = st.columns(3)
            m1.metric("Final Balance", f"${final_bal:,.0f}", delta=f"${(final_bal - start_cap):,.0f}")
            m2.metric("Total Trades", len(df_res))
            m3.metric("Win Rate", f"{win_rate:.1f}%")
            st.line_chart(df_res, x='exit_date', y='balance', color=SUCCESS_COLOR)
            st.dataframe(df_res.sort_values('exit_date', ascending=False), use_container_width=True)
        else: st.warning("No trades generated.")

# --- OPTIMIZER ---
with tab_auto:
    st.info("Uses Grid Search to find the best Entry/Exit parameters.")
    
    scan_depth = st.radio("Scan Resolution", ["Standard (36 Runs)", "Deep (144 Runs)"], horizontal=True)
    
    if 'opt_queue' not in st.session_state: st.session_state.opt_queue = []
    if 'opt_results' not in st.session_state: st.session_state.opt_results = []
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_opt = st.button("Start Optimization", type="primary")
    with col_btn2:
        stop_opt = st.button("Stop")

    if start_opt:
        if scan_depth.startswith("Standard"):
            deltas = [0.15, 0.20, 0.30]
            dtes = [21, 7, 0]
            targets = [50, 25]
            stops = [200, 300]
        else:
            deltas = [0.10, 0.15, 0.20, 0.30]
            dtes = [30, 21, 14, 7, 0]
            targets = [75, 50, 25]
            stops = [100, 200, 300]
        
        grid = list(itertools.product(deltas, dtes, targets, stops))
        st.session_state.opt_queue = grid
        st.session_state.opt_results = []
        st.session_state.is_running = True
        st.rerun()

    if stop_opt:
        st.session_state.is_running = False
        st.session_state.opt_queue = []
        st.rerun()

    if st.session_state.get('is_running', False):
        if not st.session_state.opt_queue:
            st.session_state.is_running = False
            st.rerun()
            
        cfg = st.session_state.opt_queue.pop(0)
        d_delta, d_dte, d_target, d_stop = cfg
        
        remaining = len(st.session_state.opt_queue)
        total_runs = len(st.session_state.opt_results) + remaining + 1
        curr_run = len(st.session_state.opt_results) + 1
        
        st.write(f"**Processing Config {curr_run}/{total_runs}**")
        st.caption(f"Entry: {d_delta} Delta | Exit: {d_dte} DTE | Target: {d_target}% | Stop: {d_stop}%")
        
        main_bar = st.progress(curr_run / total_runs)
        inner_bar = st.progress(0, text="Simulating Timeline...")
        
        df_r, bal = run_portfolio_simulation(
            st.session_state.data_map, d_delta, d_dte, d_stop, d_target,
            start_cap, monthly_add, slippage, fees, inner_bar
        )
        
        if not df_r.empty:
            roi = ((bal - start_cap) / start_cap) * 100
            wr = len(df_r[df_r['pnl_dollars']>0]) / len(df_r) * 100
            dd_series = (df_r['balance'] - df_r['balance'].cummax()) / df_r['balance'].cummax()
            max_dd = dd_series.min() * 100
            
            st.session_state.opt_results.append({
                "Delta": d_delta, "DTE": d_dte, "Target": d_target, "Stop": d_stop,
                "Final Balance": bal, "ROI %": roi, "Win Rate": wr, "Max DD %": max_dd
            })
        st.rerun()

    if st.session_state.opt_results and not st.session_state.get('is_running', False):
        res_df = pd.DataFrame(st.session_state.opt_results)
        res_df = res_df.sort_values("Final Balance", ascending=False)
        
        st.subheader("Leaderboard")
        best = res_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Balance", f"${best['Final Balance']:,.0f}")
        c2.metric("Config", f"Delta {best['Delta']} | {int(best['DTE'])} DTE")
        c3.metric("Win Rate", f"{best['Win Rate']:.1f}%")
        
        st.dataframe(res_df.style.format({
            "Final Balance": "${:,.0f}", "ROI %": "{:.1f}%", "Win Rate": "{:.1f}%", "Max DD %": "{:.1f}%"
        }).background_gradient(subset="Final Balance", cmap="Greens"), use_container_width=True)
