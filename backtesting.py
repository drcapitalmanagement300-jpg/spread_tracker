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
BASKET_TICKERS = ["TSLA", "NVDA", "AAPL", "AMD", "AMZN"] # High liquid names
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
    # Safe handling for arrays
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
    
    # Intrinsic value fallback for expired/zero time
    intrinsic_val = np.maximum(K - S, 0.0)
    price = np.where(T <= 0, intrinsic_val, price)
    return np.nan_to_num(price)

# --- DATA LOADING ---
@st.cache_data(ttl=3600*24)
def get_historical_data():
    data_map = {}
    
    # 1. Fetch SPY for Market Regime (The "Finder" Filter)
    try:
        spy = yf.Ticker("SPY")
        spy_df = spy.history(period="10y")
        if not spy_df.empty:
            if spy_df.index.tz is not None: spy_df.index = spy_df.index.tz_localize(None)
            spy_df.index = pd.to_datetime(spy_df.index)
            # Calculate the 200 SMA "Crash Guard"
            spy_df['SMA200'] = spy_df['Close'].rolling(window=200).mean()
            data_map['SPY'] = spy_df
    except: pass

    # 2. Fetch Tradable Tickers
    for t in BASKET_TICKERS:
        try:
            stock = yf.Ticker(t)
            df = stock.history(period="10y")
            if df.empty: continue
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index)
            
            # Calculate Volatility for Pricing
            df['Returns'] = df['Close'].pct_change()
            df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            data_map[t] = df
        except: continue
        
    return data_map

# --- PHASE 1: GENERATE VALID ENTRIES (MATCHING FINDER LOGIC) ---
def generate_valid_entries(ticker, df, spy_df):
    entries = []
    r = 0.045 # Risk free rate approximation
    width = 5.0
    min_credit = 0.70 # Finder Criteria
    
    # Pre-calculate SPY Regime mask
    spy_aligned = spy_df.reindex(df.index, method='ffill')
    is_market_safe = (spy_aligned['Close'] > spy_aligned['SMA200']).fillna(False).values
    
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    
    for i, entry_date in enumerate(dates):
        # 1. FINDER RULE: Check Market Regime
        if not is_market_safe[i]: continue

        # 2. FINDER RULE: Target 25-50 DTE (Ideal ~45)
        target_date = entry_date + timedelta(days=45)
        future_slice = df.index[i:]
        valid_exps = future_slice[future_slice >= target_date]
        
        if valid_exps.empty: continue
        expiry = valid_exps[0] # Closest to 45 days out
        
        T = (expiry - entry_date).days / 365.0
        S = close_prices[i]
        sigma = hvs[i] # Using HV as proxy for IV
        
        # 3. FINDER RULE: Strike Selection (0.75x Expected Move)
        expected_move = S * sigma * np.sqrt(T) * 0.75
        target_short = S - expected_move
        
        # Round to nearest 1.0 or 5.0 depending on price (Simple logic)
        step = 5.0 if S > 200 else 1.0
        short_k = round(target_short / step) * step
        long_k = short_k - width
        
        # 4. PRICE THE SPREAD
        p_short = vectorized_black_scholes(S, short_k, T, r, sigma)
        p_long = vectorized_black_scholes(S, long_k, T, r, sigma)
        credit = float(p_short - p_long)
        
        # 5. FINDER RULE: Min Credit Check
        if credit < min_credit: continue
        
        entries.append({
            'entry_idx': i,
            'entry_date': entry_date,
            'expiry_date': expiry,
            'short_k': short_k,
            'long_k': long_k,
            'credit': credit,
            'margin': width * 100 # Margin is width of spread * 100
        })
        
    return entries

# --- PHASE 2: SIMULATE EXITS (THE GRID SEARCH) ---
def simulate_strategy(entries, df, exit_settings):
    # Unpack settings
    stop_loss_mult = exit_settings['stop_mult'] # e.g. 3.0 for 300%
    dte_threshold = exit_settings['dte_thresh'] # e.g. 14
    profit_target = exit_settings['profit_target'] # e.g. 0.50 for 50%
    
    results = []
    r = 0.045
    
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    date_map = {d: i for i, d in enumerate(dates)}
    
    for e in entries:
        start_idx = e['entry_idx']
        expiry_idx = date_map.get(e['expiry_date'])
        if not expiry_idx: continue
        
        # Get path data
        path_S = close_prices[start_idx+1 : expiry_idx+1]
        path_sigma = hvs[start_idx+1 : expiry_idx+1]
        path_dates = dates[start_idx+1 : expiry_idx+1]
        
        if len(path_S) == 0: continue
        
        # Calculate Time Remaining for each day in path
        expiry_ts = e['expiry_date']
        path_days_left = np.array([(expiry_ts - d).days for d in path_dates])
        path_T = path_days_left / 365.0
        
        # Re-price spread daily
        curr_short = vectorized_black_scholes(path_S, e['short_k'], path_T, r, path_sigma)
        curr_long = vectorized_black_scholes(path_S, e['long_k'], path_T, r, path_sigma)
        spread_val = curr_short - curr_long
        
        # Check Triggers
        # 1. Profit Target (Value drops below X% of credit)
        # 2. Stop Loss (Value rises above X% of credit)
        # 3. DTE Limit (Days left <= X)
        
        target_val = e['credit'] * (1 - profit_target)
        stop_val = e['credit'] * stop_loss_mult
        
        mask_win = spread_val <= target_val
        mask_loss = spread_val >= stop_val
        mask_time = path_days_left <= dte_threshold
        
        # Find first trigger
        triggers = mask_win | mask_loss | mask_time
        
        if triggers.any():
            idx = np.argmax(triggers)
            exit_date = path_dates[idx]
            final_val = spread_val[idx]
            pnl = (e['credit'] - final_val) * 100 # Multiplier
            
            reason = "Expired"
            if mask_loss[idx]: reason = "Stop Loss"
            elif mask_win[idx]: reason = "Profit Target"
            elif mask_time[idx]: reason = "Time Exit"
            
            results.append({
                'exit_date': exit_date,
                'pnl': pnl,
                'margin': e['margin'],
                'reason': reason
            })
        else:
            # Held to expiration
            final_val = spread_val[-1] # Should be intrinsic
            pnl = (e['credit'] - final_val) * 100
            results.append({
                'exit_date': e['expiry_date'],
                'pnl': pnl,
                'margin': e['margin'],
                'reason': "Expiration"
            })
            
    return results

# --- UI LAYER ---
header_col1, header_col2 = st.columns([1, 6])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=100)
    except: st.write("DR CAPITAL")
with header_col2:
    st.title("Options Lab: Strategy Optimizer")
    st.caption("Historical simulation of the 'Spread Finder' logic with variable exit criteria.")

st.markdown("---")

# Initialize Data
if 'hist_data' not in st.session_state:
    with st.spinner("Loading 10 Years of Market Data..."):
        st.session_state.hist_data = get_historical_data()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Simulation Config")
    start_cap = st.number_input("Starting Capital", 1000, 1000000, 10000)
    monthly_add = st.number_input("Monthly Add", 0, 10000, 500)
    
    st.divider()
    st.subheader("Finder Criteria (Fixed)")
    st.markdown("""
    * **Regime:** SPY > 200 SMA
    * **Min Credit:** $0.70
    * **Target:** 0.75x Exp. Move
    * **Entry:** ~45 DTE
    """)
    
    if st.button("Start Grid Search", type="primary"):
        st.session_state.running = True

# --- MAIN LOGIC ---
if st.session_state.get('running', False):
    
    # 1. Pre-calculate Entries (Once per ticker)
    if 'entries_cache' not in st.session_state:
        st.info("Phase 1: Generating Valid Entries based on Finder Logic...")
        entries_cache = {}
        spy_data = st.session_state.hist_data.get('SPY')
        
        for t, df in st.session_state.hist_data.items():
            if t == 'SPY': continue
            entries_cache[t] = generate_valid_entries(t, df, spy_data)
        st.session_state.entries_cache = entries_cache
    
    # 2. Define Grid
    # Spread Value: 100% (Scratch), 200% (2x Loss), 300% (3x Loss), 400%
    stop_mults = [1.0, 2.0, 3.0, 4.0] 
    dtes = [7, 14, 21]
    targets = [0.25, 0.50, 0.75]
    
    grid = list(itertools.product(stop_mults, dtes, targets))
    total_runs = len(grid)
    
    results_summary = []
    
    prog_bar = st.progress(0)
    status = st.empty()
    
    # 3. Run Grid
    for i, (stop, dte, target) in enumerate(grid):
        status.write(f"Testing: Stop {stop*100:.0f}% | Exit {dte} DTE | Profit {target*100:.0f}%")
        prog_bar.progress((i+1)/total_runs)
        
        config_pnl = []
        config_trades = 0
        
        # Run all tickers for this config
        for t, entries in st.session_state.entries_cache.items():
            trade_results = simulate_strategy(entries, st.session_state.hist_data[t], {
                'stop_mult': stop,
                'dte_thresh': dte,
                'profit_target': target
            })
            for tr in trade_results:
                config_pnl.append(tr)
                
        # Aggregate Results
        if config_pnl:
            df_res = pd.DataFrame(config_pnl)
            total_pnl = df_res['pnl'].sum()
            win_rate = len(df_res[df_res['pnl'] > 0]) / len(df_res) * 100
            avg_win = df_res[df_res['pnl'] > 0]['pnl'].mean() if len(df_res[df_res['pnl'] > 0]) > 0 else 0
            avg_loss = df_res[df_res['pnl'] <= 0]['pnl'].mean() if len(df_res[df_res['pnl'] <= 0]) > 0 else 0
            
            # Simple Drawdown Calc
            df_res = df_res.sort_values('exit_date')
            df_res['cum_pnl'] = df_res['pnl'].cumsum()
            peak = df_res['cum_pnl'].cummax()
            dd = (df_res['cum_pnl'] - peak).min()
            
            results_summary.append({
                "Stop Loss": f"{stop*100:.0f}%",
                "Exit DTE": dte,
                "Profit Target": f"{target*100:.0f}%",
                "Total P&L": total_pnl,
                "Win Rate": win_rate,
                "Avg Win": avg_win,
                "Avg Loss": avg_loss,
                "Max Drawdown": dd,
                "_raw_df": df_res # Store for detailed view
            })
            
    st.session_state.grid_results = pd.DataFrame(results_summary).sort_values("Total P&L", ascending=False)
    st.session_state.running = False
    st.rerun()

# --- RESULTS DISPLAY ---
if 'grid_results' in st.session_state:
    res_df = st.session_state.grid_results
    
    st.success("Optimization Complete!")
    
    # 1. Leaderboard
    st.subheader("🏆 Leaderboard: Optimal Exit Settings")
    display_df = res_df.drop(columns=["_raw_df"])
    st.dataframe(display_df.style.format({
        "Total P&L": "${:,.0f}",
        "Win Rate": "{:.1f}%",
        "Avg Win": "${:,.0f}",
        "Avg Loss": "${:,.0f}",
        "Max Drawdown": "${:,.0f}"
    }).background_gradient(subset="Total P&L", cmap="Greens"), use_container_width=True)
    
    # 2. Detailed Analysis of Winner
    best = res_df.iloc[0]
    raw = best['_raw_df']
    
    st.divider()
    st.markdown(f"### Deep Dive: The Best Strategy")
    st.info(f"**Optimal Settings:** Stop Loss at **{best['Stop Loss']}** Credit Value | Exit at **{best['Exit DTE']} DTE** | Take Profit at **{best['Profit Target']}**")
    
    # Equity Curve Construction
    raw['exit_date'] = pd.to_datetime(raw['exit_date'])
    daily_pnl = raw.set_index('exit_date')['pnl'].resample('D').sum().fillna(0)
    
    # Reconstruct Account Balance
    balance = [start_cap]
    monthly_contrib_counter = 0
    curr_bal = start_cap
    
    dates = daily_pnl.index
    equity_curve = []
    
    for d in dates:
        # Add monthly contrib logic roughly
        if d.day == 1: curr_bal += monthly_add
        
        pnl = daily_pnl.loc[d]
        curr_bal += pnl
        equity_curve.append(curr_bal)
        
    eq_df = pd.DataFrame({'Date': dates, 'Equity': equity_curve}).set_index('Date')
    
    # Charts
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Account Growth (Equity Curve)**")
        st.line_chart(eq_df, color=SUCCESS_COLOR)
        
    with c2:
        st.markdown("**Monthly Returns**")
        monthly_ret = daily_pnl.resample('M').sum()
        # Color bars red/green
        st.bar_chart(monthly_ret)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    total_years = (dates[-1] - dates[0]).days / 365.25
    cagr = ((curr_bal / start_cap) ** (1/total_years) - 1) * 100
    
    m1.metric("Final Balance", f"${curr_bal:,.0f}")
    m2.metric("CAGR (Annual Return)", f"{cagr:.1f}%")
    m3.metric("Profit Factor", f"{abs(best['Avg Win']/best['Avg Loss']):.2f}")
    m4.metric("Avg Monthly Income", f"${monthly_ret.mean():,.0f}")
