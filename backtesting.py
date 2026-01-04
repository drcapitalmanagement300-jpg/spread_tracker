import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as si
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
    intrinsic_val = np.maximum(K - S, 0.0)
    price = np.where(T <= 0, intrinsic_val, price)
    return np.nan_to_num(price)

# --- DATA LOADING ---
@st.cache_data(ttl=3600*24)
def get_historical_data():
    data_map = {}
    try:
        spy = yf.Ticker("SPY")
        spy_df = spy.history(period="10y")
        if not spy_df.empty:
            if spy_df.index.tz is not None: spy_df.index = spy_df.index.tz_localize(None)
            spy_df.index = pd.to_datetime(spy_df.index)
            spy_df['SMA200'] = spy_df['Close'].rolling(window=200).mean()
            data_map['SPY'] = spy_df
    except: pass

    for t in BASKET_TICKERS:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="10y")
            if df.empty: continue
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index)
            df['Returns'] = df['Close'].pct_change()
            df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            data_map[t] = df
        except: continue
    return data_map

# --- PHASE 1: PRE-CALCULATE TRADE LIFECYCLES (THE SPEED HACK) ---
# We calculate the daily option price path ONCE per trade.
# This prevents re-running Black-Scholes inside the grid loop.
def generate_trade_lifecycles(ticker, df, spy_df):
    lifecycle_data = []
    r = 0.045
    width = 5.0
    min_credit = 0.70
    
    # Pre-calculate Regime
    spy_aligned = spy_df.reindex(df.index, method='ffill')
    is_market_safe = (spy_aligned['Close'] > spy_aligned['SMA200']).fillna(False).values
    
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    
    # Iterate through potential entry days
    for i in range(len(dates)):
        # Finder Criteria 1: Market Regime
        if not is_market_safe[i]: continue
        
        # Finder Criteria 2: Date Selection (~45 DTE)
        entry_date = dates[i]
        target_date = entry_date + timedelta(days=45)
        
        # Fast forward to find closest expiry
        future_indices = np.where(dates >= target_date)[0]
        if len(future_indices) == 0: continue
        expiry_idx = future_indices[0]
        expiry_date = dates[expiry_idx]
        
        T_entry = (expiry_date - entry_date).days / 365.0
        S_entry = close_prices[i]
        sigma_entry = hvs[i]
        
        # Finder Criteria 3: Strikes (0.75x Expected Move)
        exp_move = S_entry * sigma_entry * np.sqrt(T_entry) * 0.75
        target_short = S_entry - exp_move
        step = 5.0 if S_entry > 200 else 1.0
        short_k = round(target_short / step) * step
        long_k = short_k - width
        
        # Price Check
        p_s = vectorized_black_scholes(S_entry, short_k, T_entry, r, sigma_entry)
        p_l = vectorized_black_scholes(S_entry, long_k, T_entry, r, sigma_entry)
        credit = float(p_s - p_l)
        
        if credit < min_credit: continue
        
        # --- PRE-CALCULATE LIFECYCLE ARRAYS ---
        # We grab the price/vol/date path for the entire trade duration
        path_slice = slice(i+1, expiry_idx+1)
        path_S = close_prices[path_slice]
        path_vol = hvs[path_slice]
        path_dates = dates[path_slice]
        
        if len(path_S) == 0: continue
        
        # Vectorized Time Remaining
        path_dtes = np.array([(expiry_date - d).days for d in path_dates])
        path_T = path_dtes / 365.0
        
        # Vectorized Pricing for the whole path
        curr_short = vectorized_black_scholes(path_S, short_k, path_T, r, path_vol)
        curr_long = vectorized_black_scholes(path_S, long_k, path_T, r, path_vol)
        path_spread_values = curr_short - curr_long
        
        lifecycle_data.append({
            'credit': credit,
            'margin': width * 100,
            'dates': path_dates,       # Array of daily dates
            'values': path_spread_values, # Array of daily spread prices
            'dtes': path_dtes          # Array of daily DTEs
        })
        
    return lifecycle_data

# --- PHASE 2: INSTANT GRID SIMULATION ---
def simulate_grid_fast(all_trades, exit_config):
    stop_mult = exit_config['stop_mult']
    dte_thresh = exit_config['dte_thresh']
    profit_target = exit_config['profit_target']
    
    results = []
    
    for t in all_trades:
        credit = t['credit']
        
        # Thresholds
        target_val = credit * (1 - profit_target)
        stop_val = credit * stop_mult
        
        # Boolean Masking (Super Fast)
        # Find indices where conditions are met
        # We assume arrays are synced: dates, values, dtes
        
        # 1. Profit Trigger
        wins = t['values'] <= target_val
        # 2. Stop Trigger
        losses = t['values'] >= stop_val
        # 3. Time Trigger
        times = t['dtes'] <= dte_thresh
        
        # Combine triggers
        triggers = wins | losses | times
        
        if triggers.any():
            # Find first occurrence
            idx = np.argmax(triggers)
            
            final_val = t['values'][idx]
            exit_date = t['dates'][idx]
            pnl = (credit - final_val) * 100
            
            # Determine reason (Order matters: Loss checks usually execute first in real life stops)
            if losses[idx]: reason = "Stop"
            elif wins[idx]: reason = "Profit"
            else: reason = "Time"
            
            results.append({
                'date': exit_date,
                'pnl': pnl,
                'margin': t['margin'],
                'reason': reason
            })
        else:
            # Expired
            final_val = t['values'][-1]
            pnl = (credit - final_val) * 100
            results.append({
                'date': t['dates'][-1],
                'pnl': pnl,
                'margin': t['margin'],
                'reason': "Expired"
            })
            
    return results

# --- UI LAYER ---
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=100)
    except: st.write("DR CAPITAL")
with header_col2:
    st.title("Options Lab: Hyper-Speed Optimizer")
    st.caption("Pre-calculates Option Lifecycles to perform Grid Search instantly.")

st.markdown("---")

# 1. INIT DATA
if 'hist_data' not in st.session_state:
    with st.spinner("Loading 10 Years of Market Data..."):
        st.session_state.hist_data = get_historical_data()

# 2. SIDEBAR
with st.sidebar:
    st.header("Account Settings")
    start_cap = st.number_input("Starting Capital", 1000, 1000000, 10000)
    monthly_add = st.number_input("Monthly Add", 0, 10000, 500)
    
    st.divider()
    st.info("Finder Rules Fixed: SPY > 200SMA, Credit > $0.70, 0.75x EM.")
    
    if st.button("Run Optimizer", type="primary"):
        st.session_state.run_opt = True

# 3. MAIN EXECUTION
if st.session_state.get('run_opt', False):
    
    # A. PRE-CALCULATION (Math Phase)
    if 'lifecycle_cache' not in st.session_state:
        st.info("Phase 1: Pre-calculating Trade Lifecycles (This happens once)...")
        cache = []
        spy_df = st.session_state.hist_data.get('SPY')
        
        # Parallel Processing for Tickers
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for t, df in st.session_state.hist_data.items():
                if t == 'SPY': continue
                futures.append(executor.submit(generate_trade_lifecycles, t, df, spy_df))
            
            for f in concurrent.futures.as_completed(futures):
                cache.extend(f.result())
        
        st.session_state.lifecycle_cache = cache
        st.success(f"Phase 1 Complete. Cached {len(cache)} unique trade paths.")

    # B. GRID SEARCH (Logic Phase)
    st.write("Phase 2: Running Grid Search...")
    
    # Define Grid
    stop_mults = [1.0, 2.0, 3.0, 4.0] # 100% to 400%
    dtes = [7, 14, 21]
    targets = [0.25, 0.50, 0.75]
    
    grid = list(itertools.product(stop_mults, dtes, targets))
    total_runs = len(grid)
    prog_bar = st.progress(0)
    
    summary_list = []
    
    for i, (stop, dte, target) in enumerate(grid):
        prog_bar.progress((i+1)/total_runs)
        
        # Run Simulation
        trade_results = simulate_grid_fast(st.session_state.lifecycle_cache, {
            'stop_mult': stop,
            'dte_thresh': dte,
            'profit_target': target
        })
        
        if not trade_results: continue
        
        # Aggregation
        df_res = pd.DataFrame(trade_results)
        total_pnl = df_res['pnl'].sum()
        win_rate = (len(df_res[df_res['pnl'] > 0]) / len(df_res)) * 100
        avg_win = df_res[df_res['pnl'] > 0]['pnl'].mean() if not df_res[df_res['pnl'] > 0].empty else 0
        avg_loss = df_res[df_res['pnl'] <= 0]['pnl'].mean() if not df_res[df_res['pnl'] <= 0].empty else 0
        
        # Drawdown Calc
        df_res = df_res.sort_values('date')
        df_res['cum_pnl'] = df_res['pnl'].cumsum()
        peak = df_res['cum_pnl'].cummax()
        dd = (df_res['cum_pnl'] - peak).min()
        
        summary_list.append({
            "Stop Loss": f"{stop*100:.0f}%",
            "Exit DTE": dte,
            "Profit Target": f"{target*100:.0f}%",
            "Total P&L": total_pnl,
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Max DD": dd,
            "_raw": df_res # For deep dive
        })
        
    st.session_state.opt_results = pd.DataFrame(summary_list).sort_values("Total P&L", ascending=False)
    st.session_state.run_opt = False # Reset trigger
    st.rerun()

# 4. REPORTING
if 'opt_results' in st.session_state:
    res_df = st.session_state.opt_results
    
    st.success("Analysis Complete.")
    
    # Leaderboard
    st.subheader("🏆 Optimization Leaderboard")
    display_cols = ["Stop Loss", "Exit DTE", "Profit Target", "Total P&L", "Win Rate", "Avg Win", "Avg Loss", "Max DD"]
    st.dataframe(res_df[display_cols].head(10).style.format({
        "Total P&L": "${:,.0f}",
        "Win Rate": "{:.1f}%",
        "Avg Win": "${:,.0f}",
        "Avg Loss": "${:,.0f}",
        "Max DD": "${:,.0f}"
    }).background_gradient(subset="Total P&L", cmap="Greens"), use_container_width=True)
    
    # Best Strategy Deep Dive
    best = res_df.iloc[0]
    st.divider()
    st.header("Deep Dive: The Best Config")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Stop Loss", best['Stop Loss'])
    c2.metric("Profit Target", best['Profit Target'])
    c3.metric("Exit DTE", best['Exit DTE'])
    
    # Reconstruct Equity Curve
    raw = best['_raw']
    raw['date'] = pd.to_datetime(raw['date'])
    daily_pnl = raw.set_index('date')['pnl'].resample('D').sum().fillna(0)
    
    equity = [start_cap]
    curr = start_cap
    dates = daily_pnl.index
    
    for d in dates:
        if d.day == 1: curr += monthly_add # Monthly add logic
        curr += daily_pnl.loc[d]
        equity.append(curr)
    
    # Fix length mismatch
    eq_dates = [dates[0] - timedelta(days=1)] + list(dates)
    eq_df = pd.DataFrame({'Equity': equity}, index=eq_dates)
    
    st.subheader("Account Growth")
    st.line_chart(eq_df, color=SUCCESS_COLOR)
    
    # Monthly Stats
    m_ret = daily_pnl.resample('M').sum()
    st.subheader("Monthly Returns")
    st.bar_chart(m_ret)
    
    m1, m2, m3 = st.columns(3)
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = ((curr / start_cap) ** (1/years) - 1) * 100
    
    m1.metric("CAGR", f"{cagr:.1f}%")
    m2.metric("Avg Monthly P&L", f"${m_ret.mean():,.0f}")
    m3.metric("Win Rate", f"{best['Win Rate']:.1f}%")
