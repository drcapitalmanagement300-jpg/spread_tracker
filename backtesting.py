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

# --- DATA LOADER ---
@st.cache_data(ttl=3600*24)
def get_market_data(ticker):
    attempts = 0
    max_retries = 3
    while attempts < max_retries:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="10y") 
            if df.empty: raise ValueError("Empty Data")
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df['Returns'] = df['Close'].pct_change()
            df['HV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            return df
        except Exception as e:
            attempts += 1
            time.sleep(1)
            if attempts == max_retries:
                st.error(f"Failed to download {ticker}. Error: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

# --- SIMULATION ENGINE ---
def run_simulation(df, entry_delta, exit_delta_trigger, exit_spread_pct, exit_dte_trigger, profit_target_pct, slippage, fees, capital):
    if df.empty: return pd.DataFrame()

    all_potential_trades = []
    entry_dates = df.index
    r = 0.045
    
    # FAST LOOP (No progress bar needed for optimizer chunks)
    for i, entry_date in enumerate(entry_dates):
        try:
            entry_row = df.loc[entry_date]
            S_entry = float(entry_row['Close'])
            sigma_entry = float(entry_row['HV'])
            
            target_expiry = entry_date + timedelta(days=45)
            future_data = df.loc[entry_date:]
            valid_expiries = future_data.index[future_data.index >= target_expiry]
            
            if valid_expiries.empty: continue
            expiry_date = valid_expiries[0]
            T_entry = (expiry_date - entry_date).days / 365.0
            
            # Strike Selection (Optimized Search)
            best_strike = S_entry * 0.8
            min_delta_diff = 1.0
            
            # Low resolution scan for speed
            scan_strikes = np.linspace(S_entry * 0.70, S_entry, 20) 
            
            for k in scan_strikes:
                _, d = black_scholes_put(S_entry, k, T_entry, r, sigma_entry)
                diff = abs(abs(d) - abs(entry_delta))
                if diff < min_delta_diff: min_delta_diff = diff; best_strike = k

            strike_step = 5.0 if S_entry > 200 else 1.0
            short_strike = round(best_strike / strike_step) * strike_step
            long_strike = short_strike - 5.0
            
            p_s, d_s = black_scholes_put(S_entry, short_strike, T_entry, r, sigma_entry)
            p_l, _ = black_scholes_put(S_entry, long_strike, T_entry, r, sigma_entry)
            
            gross_credit = p_s - p_l
            if gross_credit < 0.10: continue 
            
            trade_res = {
                'entry_date': entry_date,
                'short_strike': short_strike,
                'gross_credit': gross_credit,
                'exit_reason': 'Held to Expiry',
                'pnl': gross_credit,
                'exit_date': expiry_date
            }
            
            # Walk Forward
            trade_path = df.loc[entry_date:expiry_date].iloc[1:]
            
            for curr_date, row in trade_path.iterrows():
                S_curr = float(row['Close'])
                T_curr = (expiry_date - curr_date).days / 365.0
                days_left = (expiry_date - curr_date).days
                sigma_curr = float(row['HV'])
                
                curr_p_s, curr_d_s = black_scholes_put(S_curr, short_strike, T_curr, r, sigma_curr)
                curr_p_l, _ = black_scholes_put(S_curr, long_strike, T_curr, r, sigma_curr)
                spread_val = curr_p_s - curr_p_l
                
                profit_captured_pct = ((gross_credit - spread_val) / gross_credit) * 100
                loss_pct = (spread_val / gross_credit) * 100
                
                hit_exit = False
                reason = ""
                
                if days_left < exit_dte_trigger: hit_exit = True; reason = f"DTE < {exit_dte_trigger}"
                elif abs(curr_d_s) > exit_delta_trigger: hit_exit = True; reason = f"Delta > {exit_delta_trigger}"
                elif loss_pct > exit_spread_pct: hit_exit = True; reason = f"Max Loss"
                elif profit_captured_pct >= profit_target_pct: hit_exit = True; reason = f"Profit Target"
                
                if hit_exit:
                    trade_res['exit_reason'] = reason
                    trade_res['pnl'] = gross_credit - spread_val
                    trade_res['exit_date'] = curr_date
                    break
            
            all_potential_trades.append(trade_res)
        except: continue
    
    # CAPITAL CONSTRAINTS
    if not all_potential_trades: return pd.DataFrame(), 0
    
    friction = (slippage * 4) + fees 
    executed_trades = []
    active_end_dates = [] 
    max_positions = int(capital // 500) 
    skipped_count = 0
    
    for trade in all_potential_trades:
        entry = trade['entry_date']
        active_end_dates = [d for d in active_end_dates if d > entry]
        
        if len(active_end_dates) < max_positions:
            trade['pnl'] = trade['pnl'] - (friction / 100.0)
            executed_trades.append(trade)
            active_end_dates.append(trade['exit_date'])
        else:
            skipped_count += 1
            
    results = pd.DataFrame(executed_trades)
    return results, skipped_count

# --- UI HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""<div style='text-align: left; padding-top: 10px;'><h1 style='margin-bottom: 0px;'>Options Lab</h1><p style='color: gray;'>Portfolio Simulation & Optimizer</p></div>""", unsafe_allow_html=True)
with header_col3: st.write("") 
st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- GLOBAL SETTINGS ---
with st.sidebar:
    st.header("Global Settings")
    ticker = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "GLD", "NVDA", "AAPL", "AMD", "TSLA", "MSFT", "AMZN"], index=0)
    
    if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
        with st.spinner(f"Fetching 10y Data for {ticker}..."):
            st.session_state.df_market = get_market_data(ticker)
            st.session_state.last_ticker = ticker
            
    st.markdown("---")
    st.caption("Account Parameters")
    start_cap = st.number_input("Capital ($)", 1000, 1000000, 10000, 1000)
    slippage = st.number_input("Slippage ($/leg)", 0.00, 0.10, 0.01, 0.01)
    fees = st.number_input("Fees ($/trade)", 0.00, 5.00, 0.00, 0.10)

# --- TABS ---
tab_sim, tab_opt = st.tabs(["🧪 Manual Lab", "🤖 Auto-Optimizer"])

# ==========================================
# TAB 1: MANUAL LAB
# ==========================================
with tab_sim:
    c1, c2, c3, c4 = st.columns(4)
    with c1: exit_delta = st.number_input("Exit Short Delta >", 0.1, 1.0, 0.60, 0.05)
    with c2: exit_spread_pct = st.number_input("Exit Spread Value % >", 100, 1000, 300, 50)
    with c3: exit_dte = st.number_input("Exit DTE <", 0, 30, 21)
    with c4: profit_target = st.number_input("Take Profit %", 10, 90, 50, 5)

    if st.button("Run Manual Simulation", type="primary", use_container_width=True):
        if 'df_market' in st.session_state:
            with st.spinner("Simulating..."):
                results, skipped = run_simulation(
                    st.session_state.df_market, 0.20, exit_delta, exit_spread_pct, exit_dte, profit_target,
                    slippage, fees, start_cap
                )
            
            if not results.empty:
                total_pnl = results['pnl'].sum() * 100
                win_rate = len(results[results['pnl']>0]) / len(results) * 100
                st.success(f"**Net Profit:** ${total_pnl:,.0f} | **Win Rate:** {win_rate:.1f}%")
                
                # Charts
                results['cum_pnl'] = results['pnl'].cumsum() * 100
                results['equity'] = start_cap + results['cum_pnl']
                results['date'] = pd.to_datetime(results['entry_date'])
                results = results.sort_values('date')
                
                st.line_chart(results, x='date', y='equity', color=SUCCESS_COLOR)
                
                with st.expander("View Trade Log"):
                    st.dataframe(results[['display_entry', 'display_exit', 'gross_credit', 'exit_reason', 'pnl']], use_container_width=True)
            else:
                st.warning("No trades found.")

# ==========================================
# TAB 2: OPTIMIZER (Chunked Processing)
# ==========================================
with tab_opt:
    st.markdown("### 🔍 Find the Best Settings")
    st.info("Uses 'Chunked Processing' to prevent timeouts. The page will reload automatically as it works.")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        scan_mode = st.radio("Scan Intensity", ["Light (12 Runs)", "Deep (100+ Runs)"], horizontal=True)
    
    # 1. Start Button: Initializes the Queue
    if st.button("🚀 Start Optimizer", type="primary"):
        if 'df_market' in st.session_state:
            # Define Parameter Grid
            if scan_mode.startswith("Light"):
                param_grid = {
                    "exit_dte": [21, 7],
                    "profit_target": [50, 25],
                    "stop_loss": [200, 300, 400]
                }
            else:
                param_grid = {
                    "exit_dte": [21, 14, 7, 0],
                    "profit_target": [75, 50, 25],
                    "stop_loss": [200, 300, 400, 500],
                    "exit_delta": [0.4, 0.6] 
                }
            
            # Generate Permutations
            keys, values = zip(*param_grid.items())
            permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            # Save to Session State (The "Queue")
            st.session_state.opt_queue = permutations
            st.session_state.opt_results = []
            st.session_state.opt_total = len(permutations)
            st.session_state.is_optimizing = True
            st.rerun()

    # 2. Worker Logic: Processes A BATCH and Reruns
    if st.session_state.get('is_optimizing', False):
        
        # Check if queue is empty
        if not st.session_state.opt_queue:
            st.session_state.is_optimizing = False
            st.rerun()
            
        # UI Progress
        total = st.session_state.opt_total
        remaining = len(st.session_state.opt_queue)
        done = total - remaining
        
        progress = done / total
        st.write(f"⚙️ **Processing Batch... ({done}/{total})**")
        st.progress(progress)
        
        # PROCESS BATCH (5 at a time)
        BATCH_SIZE = 5
        
        for _ in range(BATCH_SIZE):
            if not st.session_state.opt_queue:
                break
                
            # Pop the next item
            p = st.session_state.opt_queue.pop(0)
            
            # Defaults
            d_delta = p.get('exit_delta', 0.60)
            d_stop = p.get('stop_loss', 300)
            d_dte = p.get('exit_dte', 21)
            d_target = p.get('profit_target', 50)
            
            # Run Simulation
            res, _ = run_simulation(
                st.session_state.df_market, 
                0.20, d_delta, d_stop, d_dte, d_target,
                slippage, fees, start_cap
            )
            
            # Save Result
            if not res.empty:
                pnl = res['pnl'].sum() * 100
                wr = len(res[res['pnl']>0])/len(res)*100
                
                res['equity'] = start_cap + (res['pnl'].cumsum() * 100)
                res['peak'] = res['equity'].cummax()
                res['dd'] = (res['equity'] - res['peak']) / res['peak']
                max_dd = res['dd'].min() * 100
                
                st.session_state.opt_results.append({
                    "DTE": d_dte,
                    "Target %": d_target,
                    "Stop %": d_stop,
                    "Net Profit": pnl,
                    "Win Rate": wr,
                    "Max DD": max_dd,
                    "Trades": len(res)
                })
        
        # RERUN AFTER BATCH
        st.rerun()

    # 3. Results Display (When Finished)
    if not st.session_state.get('is_optimizing', False) and 'opt_results' in st.session_state and st.session_state.opt_results:
        st.balloons()
        st.subheader("🏆 Optimization Results")
        
        df_opt = pd.DataFrame(st.session_state.opt_results)
        if not df_opt.empty:
            df_opt = df_opt.sort_values("Net Profit", ascending=False)
            
            best = df_opt.iloc[0]
            b1, b2, b3 = st.columns(3)
            b1.metric("Best Profit", f"${best['Net Profit']:,.0f}")
            b2.metric("Best Config", f"Ex: {int(best['DTE'])} DTE | TP: {int(best['Target %'])}%")
            b3.metric("Win Rate", f"{best['Win Rate']:.1f}%")
            
            st.dataframe(
                df_opt.style.format({
                    "Net Profit": "${:,.0f}", 
                    "Win Rate": "{:.1f}%", 
                    "Max DD": "{:.1f}%"
                }).background_gradient(subset=["Net Profit"], cmap="Greens"),
                use_container_width=True
            )
            
            if st.button("Clear Results"):
                del st.session_state.opt_results
                st.rerun()
