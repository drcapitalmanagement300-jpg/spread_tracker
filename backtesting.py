import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import scipy.stats as si
import time
import itertools
import os
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
BACKUP_FILE = "optimization_results.csv"

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
    S = np.array(S, dtype=float)
    K = np.array(K, dtype=float)
    T = np.array(T, dtype=float)
    sigma = np.array(sigma, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    
    nd1 = si.norm.cdf(-d1)
    nd2 = si.norm.cdf(-d2)
    
    price = (K * np.exp(-r * T) * nd2) - (S * nd1)
    delta = nd1 - 1
    
    intrinsic_val = np.maximum(K - S, 0.0)
    price = np.where(T <= 0, intrinsic_val, price)
    
    price = np.nan_to_num(price)
    delta = np.nan_to_num(delta)
    
    return price, delta

# --- DATA LOADING ---
@st.cache_data(ttl=3600*24)
def get_basket_data(tickers):
    data_map = {}
    
    try:
        spy = yf.Ticker("SPY")
        spy_df = spy.history(period="10y")
        if not spy_df.empty:
            if spy_df.index.tz is not None: 
                spy_df.index = spy_df.index.tz_localize(None)
            spy_df.index = pd.to_datetime(spy_df.index) 
            
            spy_df['MA100'] = spy_df['Close'].rolling(window=100).mean()
            spy_df['MA200'] = spy_df['Close'].rolling(window=200).mean()
            spy_df['MA300'] = spy_df['Close'].rolling(window=300).mean()
            spy_df['MA400'] = spy_df['Close'].rolling(window=400).mean()
            data_map['SPY'] = spy_df
    except: pass

    for t in tickers:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="10y")
            if df.empty: continue
            
            if df.index.tz is not None: 
                df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index)
            
            df['Returns'] = df['Close'].pct_change()
            df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            data_map[t] = df
        except: continue
        
    return data_map

# --- PHASE 1: PRE-CALCULATE ENTRIES (THE SPEED HACK) ---
# We calculate the "Entry Candidate" once per Delta/SMA combo and cache it.
# This skips the expensive strike search loop during the main simulation.
def precalculate_entries(ticker, df, spy_df, entry_delta, sma_filter_type):
    entries = []
    r = 0.045
    strike_steps = 30
    
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    
    # Crash Guard Check
    market_safe = np.full(len(dates), True, dtype=bool) 
    if sma_filter_type != "None" and not spy_df.empty:
        try:
            aligned_spy = spy_df.reindex(dates, method='ffill')
            if sma_filter_type in aligned_spy.columns:
                 market_safe = (aligned_spy['Close'] > aligned_spy[sma_filter_type]).fillna(False).values
        except: pass
            
    for i, entry_date in enumerate(dates):
        if not market_safe[i]: continue

        S = close_prices[i]
        sigma = hvs[i]
        
        # Target 45 DTE
        target_date = entry_date + timedelta(days=45)
        future_slice = df.index[i:]
        valid_exps = future_slice[future_slice >= target_date]
        if valid_exps.empty: continue
        expiry = valid_exps[0]
        
        T = (expiry - entry_date).days / 365.0
        
        # Strike Search
        candidates = np.linspace(S * 0.50, S, strike_steps) 
        _, d_arr = vectorized_black_scholes(
            np.full(strike_steps, S), candidates, np.full(strike_steps, T), r, np.full(strike_steps, sigma)
        )
        best_idx = np.argmin(np.abs(np.abs(d_arr) - abs(entry_delta)))
        best_strike = candidates[best_idx]
        
        step = 5.0 if S > 200 else 1.0
        short_k = round(best_strike / step) * step
        long_k = short_k - 5.0
        
        p_s, _ = vectorized_black_scholes(S, short_k, T, r, sigma)
        p_l, _ = vectorized_black_scholes(S, long_k, T, r, sigma)
        credit = float(p_s - p_l)
        
        if credit < 0.10: continue 

        entries.append({
            'entry_idx': i,
            'entry_date': entry_date,
            'expiry_date': expiry,
            'short_k': short_k,
            'long_k': long_k,
            'credit': credit,
            'margin': 500.0,
            'T_initial': T
        })
    return entries

# --- PHASE 2: FAST EXIT SIMULATION ---
def simulate_exits_for_entries(entries, df, exit_dte, stop_loss_pct, profit_target_pct):
    signals = []
    r = 0.045
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    # Map dates to integer indices for O(1) lookup
    date_map = {d: i for i, d in enumerate(dates)}
    
    for e in entries:
        try:
            start_idx = e['entry_idx']
            expiry_idx = date_map.get(e['expiry_date'])
            if not expiry_idx: continue
            
            # Slice the path
            path_S = close_prices[start_idx+1 : expiry_idx+1]
            path_sigma = hvs[start_idx+1 : expiry_idx+1]
            path_dates = dates[start_idx+1 : expiry_idx+1]
            
            if len(path_S) == 0: continue
            
            # Vectorized Path Calculation
            expiry_ts = e['expiry_date']
            path_days_left = np.array([(expiry_ts - d).days for d in path_dates])
            path_T = path_days_left / 365.0
            
            path_ps, _ = vectorized_black_scholes(path_S, e['short_k'], path_T, r, path_sigma)
            path_pl, _ = vectorized_black_scholes(path_S, e['long_k'], path_T, r, path_sigma)
            path_spreads = path_ps - path_pl
            
            profit_pcts = ((e['credit'] - path_spreads) / e['credit']) * 100
            loss_pcts = (path_spreads / e['credit']) * 100
            
            mask_dte = path_days_left <= exit_dte
            mask_profit = profit_pcts >= profit_target_pct
            mask_stop = loss_pcts >= stop_loss_pct
            
            triggers = mask_dte | mask_profit | mask_stop
            
            exit_date = e['expiry_date']
            exit_reason = "Expired"
            final_pnl = e['credit']
            
            if triggers.any():
                first_hit_idx = np.argmax(triggers)
                exit_date = path_dates[first_hit_idx]
                final_pnl = e['credit'] - path_spreads[first_hit_idx]
                
                if mask_stop[first_hit_idx]: exit_reason = "Stop Loss"
                elif mask_profit[first_hit_idx]: exit_reason = "Profit Target"
                elif mask_dte[first_hit_idx]: exit_reason = f"DTE {exit_dte}"
            
            signals.append({
                'entry_date': e['entry_date'],
                'exit_date': exit_date,
                'gross_pnl': float(final_pnl),
                'exit_reason': exit_reason,
                'margin': e['margin']
            })
        except: continue
    return signals

# --- PORTFOLIO MANAGER ---
def run_portfolio_simulation(data_map, entry_cache, 
                           exit_dte, stop_loss, profit_target, 
                           start_cap, monthly_add, slippage, fees):
    
    # 1. PROCESS EXITS (FAST)
    all_signals = []
    # entry_cache is a dict: {ticker: [list of precalculated entries]}
    
    for ticker, entries in entry_cache.items():
        if ticker not in data_map: continue
        # Only simulate exits for the pre-calculated entries
        sigs = simulate_exits_for_entries(entries, data_map[ticker], exit_dte, stop_loss, profit_target)
        for s in sigs: s['ticker'] = ticker
        all_signals.extend(sigs)
            
    all_signals.sort(key=lambda x: x['entry_date'])
    
    # 2. RUN TIMELINE
    if not all_signals: return pd.DataFrame(), pd.DataFrame(), 0.0
    
    start_date = all_signals[0]['entry_date']
    end_date = all_signals[-1]['exit_date']
    timeline = pd.date_range(start=start_date, end=end_date, freq='D')
    
    current_cash = start_cap
    active_trades = []
    completed_trades = []
    equity_curve = [] # Track daily balance
    
    signal_idx = 0
    total_signals = len(all_signals)
    days_since_contrib = 0
    
    for today in timeline:
        days_since_contrib += 1
        if days_since_contrib >= 30:
            current_cash += monthly_add
            days_since_contrib = 0
            
        # Check Exits
        still_active = []
        for t in active_trades:
            if t['exit_date'] <= today:
                friction = (slippage * 4) + fees
                net_pnl = t['gross_pnl'] - (friction / 100.0)
                realized_dollar_pnl = net_pnl * 100
                current_cash += (t['margin'] + realized_dollar_pnl)
                
                completed_trades.append({
                    'exit_date': t['exit_date'],
                    'pnl_dollars': realized_dollar_pnl,
                    'balance': current_cash
                })
            else:
                still_active.append(t)
        active_trades = still_active
        
        # Check Entries
        while signal_idx < total_signals:
            sig = all_signals[signal_idx]
            if sig['entry_date'] > today: break
            if sig['entry_date'] == today:
                if current_cash >= sig['margin']:
                    current_cash -= sig['margin']
                    active_trades.append(sig)
            signal_idx += 1
        
        # Record Equity (Cash + Margin Locked)
        margin_locked = sum(t['margin'] for t in active_trades)
        equity_curve.append({'Date': today, 'Equity': current_cash + margin_locked})
            
    return pd.DataFrame(completed_trades), pd.DataFrame(equity_curve), current_cash

# --- HEADER UI ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""<div style='text-align: left; padding-top: 10px;'><h1 style='margin-bottom: 0px;'>Options Lab</h1><p style='color: gray;'>Portfolio Simulation & Optimizer</p></div>""", unsafe_allow_html=True)
with header_col3: st.write("") 
st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

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
    with st.spinner("Initializing Market Data..."):
        st.session_state.data_map = get_basket_data(BASKET_TICKERS)

# --- MANUAL LAB ---
with tab_manual:
    c1, c2 = st.columns(2)
    with c1: e_delta = st.number_input("Entry Delta", 0.1, 0.5, 0.20, 0.05)
    with c2: sma_mode = st.selectbox("Crash Guard", ["None", "MA100", "MA200", "MA300", "MA400"], index=2)
    
    c3, c4, c5 = st.columns(3)
    with c3: stop_loss = st.number_input("Stop Loss %", 100, 500, 300, 50)
    with c4: profit_target = st.number_input("Profit Target %", 10, 90, 50, 10)
    with c5: exit_dte = st.number_input("Exit DTE", 0, 30, 21)
    
    if st.button("Run Simulation", type="primary"):
        # Manual run: Pre-calculate just this one config on the fly
        cache = {}
        for t, df in st.session_state.data_map.items():
            if t == 'SPY': continue
            cache[t] = precalculate_entries(t, df, st.session_state.data_map.get('SPY'), e_delta, sma_mode)
            
        df_res, df_eq, final_bal = run_portfolio_simulation(
            st.session_state.data_map, cache,
            exit_dte, stop_loss, profit_target,
            start_cap, monthly_add, slippage, fees
        )
        
        if not df_res.empty:
            wins = df_res[df_res['pnl_dollars'] > 0]
            wr = len(wins) / len(df_res) * 100
            
            # Sharpe Ratio
            if not df_eq.empty:
                df_eq['Returns'] = df_eq['Equity'].pct_change()
                sharpe = (df_eq['Returns'].mean() / df_eq['Returns'].std()) * np.sqrt(252) if df_eq['Returns'].std() != 0 else 0
            else: sharpe = 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balance", f"${final_bal:,.0f}", f"${final_bal-start_cap:,.0f}")
            c2.metric("Win Rate", f"{wr:.1f}%")
            c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            c4.metric("Total Trades", len(df_res))
            
            st.area_chart(df_eq.set_index("Date")['Equity'], color=SUCCESS_COLOR)

# --- OPTIMIZER (OPTIMIZED & ROBUST) ---
with tab_auto:
    st.info("Performance Optimized: Caches Entries to speed up simulation by 5x-10x.")
    
    scan_depth = st.radio("Scan Mode", ["Standard", "Deep"], horizontal=True)
    
    if scan_depth.startswith("Standard"):
        smas = ["MA200", "None"]
        deltas = [0.20, 0.30]
        dtes = [21, 14]
        targets = [50]
        stops = [200, 300]
    else:
        smas = ["MA100", "MA200", "MA300", "MA400", "None"]
        deltas = [0.15, 0.20, 0.30, 0.40, 0.50]
        dtes = [21, 14]
        targets = [75, 50, 25]
        stops = [200, 300, 400, 500]
        
    grid = list(itertools.product(smas, deltas, dtes, targets, stops))
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1: start_opt = st.button("🚀 Start Optimization", type="primary")
    with col_btn2: 
        if st.button("Load Previous Results") and os.path.exists(BACKUP_FILE):
             st.session_state.opt_results_df = pd.read_csv(BACKUP_FILE)

    live_table = st.empty()
    progress_bar = st.empty()
    
    if start_opt:
        results_list = []
        
        # --- PHASE 1: PRE-CALCULATE ALL ENTRY POINTS ---
        # We identify unique (SMA, Delta) pairs to avoid redundant math
        unique_entries_needed = set((g[0], g[1]) for g in grid)
        entry_cache_global = {} # Key: (SMA, Delta) -> Value: {Ticker: [Entries]}
        
        st.write(f"Phase 1: Pre-calculating Entry Signals for {len(unique_entries_needed)} unique market setups...")
        
        # Parallel Entry Generation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_map = {}
            for (sma, delta) in unique_entries_needed:
                entry_cache_global[(sma, delta)] = {}
                for t, df in st.session_state.data_map.items():
                    if t == 'SPY': continue
                    # Submit job
                    f = executor.submit(precalculate_entries, t, df, st.session_state.data_map.get('SPY'), delta, sma)
                    future_map[f] = (sma, delta, t)
            
            for f in concurrent.futures.as_completed(future_map):
                sma, delta, t = future_map[f]
                entry_cache_global[(sma, delta)][t] = f.result()
                
        st.success("Phase 1 Complete. Starting Fast Simulations...")
        
        # --- PHASE 2: FAST SIMULATION LOOP ---
        total_runs = len(grid)
        
        for i, cfg in enumerate(grid):
            d_sma, d_delta, d_dte, d_target, d_stop = cfg
            progress_bar.progress((i + 1) / total_runs, text=f"Simulating Run {i+1}/{total_runs}")
            
            # Retrieve cached entries
            current_cache = entry_cache_global[(d_sma, d_delta)]
            
            df_r, df_eq, bal = run_portfolio_simulation(
                st.session_state.data_map, current_cache,
                d_dte, d_stop, d_target,
                start_cap, monthly_add, slippage, fees
            )
            
            if not df_r.empty:
                # Analytics
                wins = df_r[df_r['pnl_dollars'] > 0]
                losses = df_r[df_r['pnl_dollars'] <= 0]
                
                avg_win = wins['pnl_dollars'].mean() if not wins.empty else 0
                avg_loss = losses['pnl_dollars'].mean() if not losses.empty else 0
                win_rate = len(wins) / len(df_r) * 100
                
                # Sharpe & Sortino
                sharpe = 0
                sortino = 0
                if not df_eq.empty:
                    df_eq['Returns'] = df_eq['Equity'].pct_change()
                    mean_ret = df_eq['Returns'].mean()
                    std_dev = df_eq['Returns'].std()
                    if std_dev != 0: sharpe = (mean_ret / std_dev) * np.sqrt(252)
                    
                    neg_vol = df_eq[df_eq['Returns'] < 0]['Returns'].std()
                    if neg_vol != 0: sortino = (mean_ret / neg_vol) * np.sqrt(252)
                
                # Max Drawdown
                dd_series = (df_eq['Equity'] - df_eq['Equity'].cummax()) / df_eq['Equity'].cummax()
                max_dd = dd_series.min() * 100
                
                # Profit Factor
                gross_win = wins['pnl_dollars'].sum()
                gross_loss = abs(losses['pnl_dollars'].sum())
                pf = gross_win / gross_loss if gross_loss != 0 else 0
                
                res = {
                    "Crash Guard": d_sma, "Entry Delta": d_delta, 
                    "Exit DTE": d_dte, "Target %": d_target, "Stop %": d_stop,
                    "Final Balance": bal, "Sharpe": sharpe, "Profit Factor": pf,
                    "Win Rate": win_rate, "Avg Win": avg_win, "Avg Loss": avg_loss,
                    "Max DD %": max_dd
                }
                results_list.append(res)
                
                if i % 5 == 0:
                    temp_df = pd.DataFrame(results_list).sort_values("Final Balance", ascending=False)
                    live_table.dataframe(temp_df.head(5), use_container_width=True)
                    temp_df.to_csv(BACKUP_FILE, index=False)

        final_df = pd.DataFrame(results_list).sort_values("Final Balance", ascending=False)
        final_df.to_csv(BACKUP_FILE, index=False)
        st.session_state.opt_results_df = final_df
        progress_bar.empty()

    # --- REPORT ---
    if 'opt_results_df' in st.session_state:
        res_df = st.session_state.opt_results_df
        best = res_df.iloc[0]
        
        st.divider()
        st.markdown(f"###Best Config: {best['Crash Guard']} | Delta {best['Entry Delta']} | Target {best['Target %']}%")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Equity", f"${best['Final Balance']:,.0f}")
        m2.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
        m3.metric("Profit Factor", f"{best['Profit Factor']:.2f}")
        m4.metric("Max Drawdown", f"{best['Max DD %']:.1f}%")
        
        # Re-run best to get Equity Curve
        cache = {}
        for t, df in st.session_state.data_map.items():
            if t == 'SPY': continue
            cache[t] = precalculate_entries(t, df, st.session_state.data_map.get('SPY'), best['Entry Delta'], best['Crash Guard'])
            
        _, df_best_eq, _ = run_portfolio_simulation(
            st.session_state.data_map, cache,
            best['Exit DTE'], best['Stop %'], best['Target %'],
            start_cap, monthly_add, slippage, fees
        )
        
        st.subheader("Equity Curve")
        st.area_chart(df_best_eq.set_index("Date")['Equity'], color=SUCCESS_COLOR)
        
        st.subheader("Full Leaderboard")
        st.dataframe(res_df.style.format({
            "Final Balance": "${:,.0f}", "Sharpe": "{:.2f}", "Profit Factor": "{:.2f}",
            "Win Rate": "{:.1f}%", "Avg Win": "${:,.0f}", "Avg Loss": "${:,.0f}", "Max DD %": "{:.1f}%"
        }).background_gradient(subset="Final Balance", cmap="Greens"), use_container_width=True)
