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

# --- DATA LOADING (STRICT DATE FIX) ---
@st.cache_data(ttl=3600*24)
def get_basket_data(tickers):
    data_map = {}
    
    # 1. Fetch SPY for Market Filter
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

    # 2. Fetch Basket Tickers
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

# --- CORE SIGNAL GENERATOR ---
def generate_signals_for_ticker(ticker, df, spy_df, 
                                entry_delta, sma_filter_type, 
                                exit_dte, stop_loss_pct, profit_target_pct, 
                                scan_resolution_low=False):
    signals = []
    r = 0.045
    strike_steps = 10 if scan_resolution_low else 30
    
    close_prices = df['Close'].values
    hvs = df['HV'].values
    dates = df.index
    date_map = {d: i for i, d in enumerate(dates)}
    
    # --- CRASH GUARD LOGIC ---
    market_safe = np.full(len(dates), True, dtype=bool) # Default safe
    
    if sma_filter_type != "None" and not spy_df.empty:
        try:
            aligned_spy = spy_df.reindex(dates, method='ffill')
            if sma_filter_type == "MA100":
                market_safe = (aligned_spy['Close'] > aligned_spy['MA100']).fillna(False).values
            elif sma_filter_type == "MA200":
                market_safe = (aligned_spy['Close'] > aligned_spy['MA200']).fillna(False).values
            elif sma_filter_type == "MA300":
                market_safe = (aligned_spy['Close'] > aligned_spy['MA300']).fillna(False).values
            elif sma_filter_type == "MA400":
                market_safe = (aligned_spy['Close'] > aligned_spy['MA400']).fillna(False).values
        except Exception: pass
            
    for i, entry_date in enumerate(dates):
        try:
            # 1. APPLY FILTER
            if not market_safe[i]:
                continue

            S = close_prices[i]
            sigma = hvs[i]
            
            # 2. ENTRY (45 DTE)
            target_date = entry_date + timedelta(days=45)
            
            future_slice = df.index[i:]
            valid_exps = future_slice[future_slice >= target_date]
            
            if valid_exps.empty: continue
            expiry = valid_exps[0]
            expiry_idx = date_map[expiry]
            
            T = (expiry - entry_date).days / 365.0
            
            # 3. STRIKE SELECTION
            candidates = np.linspace(S * 0.50, S, strike_steps) # Widened scan range for low deltas
            _, d_arr = vectorized_black_scholes(
                np.full(strike_steps, S), 
                candidates, 
                np.full(strike_steps, T), 
                r, 
                np.full(strike_steps, sigma)
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

            # 4. EXIT LOGIC
            path_S = close_prices[i+1 : expiry_idx+1]
            path_sigma = hvs[i+1 : expiry_idx+1]
            path_dates = dates[i+1 : expiry_idx+1]
            
            if len(path_S) == 0: continue
            
            path_days_left = np.array([(expiry - d).days for d in path_dates])
            path_T = path_days_left / 365.0
            
            path_ps, _ = vectorized_black_scholes(path_S, short_k, path_T, r, path_sigma)
            path_pl, _ = vectorized_black_scholes(path_S, long_k, path_T, r, path_sigma)
            path_spreads = path_ps - path_pl
            
            profit_pcts = ((credit - path_spreads) / credit) * 100
            loss_pcts = (path_spreads / credit) * 100
            
            mask_dte = path_days_left <= exit_dte
            mask_profit = profit_pcts >= profit_target_pct
            mask_stop = loss_pcts >= stop_loss_pct
            
            triggers = mask_dte | mask_profit | mask_stop
            
            exit_date = expiry
            exit_reason = "Expired"
            final_pnl = credit
            
            if triggers.any():
                first_hit_idx = np.argmax(triggers)
                exit_date = path_dates[first_hit_idx]
                final_pnl = credit - path_spreads[first_hit_idx]
                
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
def run_portfolio_simulation(data_map, 
                           entry_delta, sma_filter_type,
                           exit_dte, stop_loss, profit_target, 
                           start_cap, monthly_add, slippage, fees, 
                           progress_bar_slot=None):
    
    spy_df = data_map.get('SPY', pd.DataFrame())
    
    # 1. GENERATE SIGNALS
    all_signals = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ticker, df in data_map.items():
            if ticker == 'SPY': continue
            futures.append(executor.submit(
                generate_signals_for_ticker, 
                ticker, df, spy_df, 
                entry_delta, sma_filter_type, 
                exit_dte, stop_loss, profit_target, 
                True
            ))
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
    with st.spinner("Initializing Market Data (Top 5 Tickers + SPY)..."):
        st.session_state.data_map = get_basket_data(BASKET_TICKERS)

# --- MANUAL LAB ---
with tab_manual:
    st.subheader("Finder Settings")
    c1, c2 = st.columns(2)
    with c1: e_delta = st.number_input("Entry Delta", 0.1, 0.5, 0.20, 0.05)
    with c2: sma_mode = st.selectbox("Crash Guard Filter", ["None", "MA100", "MA200", "MA300", "MA400"], index=2)
    
    st.subheader("Exit Settings")
    c3, c4, c5 = st.columns(3)
    with c3: stop_loss = st.number_input("Stop Loss %", 100, 500, 300, 50)
    with c4: profit_target = st.number_input("Profit Target %", 10, 90, 50, 10)
    with c5: exit_dte = st.number_input("Exit DTE", 0, 30, 21)
    
    if st.button("Run Portfolio Simulation", type="primary"):
        prog_bar = st.progress(0, text="Initializing...")
        
        df_res, final_bal = run_portfolio_simulation(
            st.session_state.data_map, 
            e_delta, sma_mode,
            exit_dte, stop_loss, profit_target,
            start_cap, monthly_add, slippage, fees, prog_bar
        )
        prog_bar.progress(1.0, text="Done!")
        time.sleep(0.5)
        prog_bar.empty()
        
        if not df_res.empty:
            wins = df_res[df_res['pnl_dollars'] > 0]['pnl_dollars']
            losses = df_res[df_res['pnl_dollars'] <= 0]['pnl_dollars']
            avg_win = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            win_rate = len(wins) / len(df_res) * 100
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Balance", f"${final_bal:,.0f}", delta=f"${(final_bal - start_cap):,.0f}")
            m2.metric("Total Trades", len(df_res))
            m3.metric("Win Rate", f"{win_rate:.1f}%")
            m4.metric("Avg Win / Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")
            
            st.line_chart(df_res, x='exit_date', y='balance', color=SUCCESS_COLOR)
            st.dataframe(df_res.sort_values('exit_date', ascending=False), use_container_width=True)
        else: st.warning("No trades generated.")

# --- OPTIMIZER ---
with tab_auto:
    st.info("Uses ~600 permutations to find the exact 'Sweet Spot' for finding and exiting trades.")
    
    scan_depth = st.radio("Scan Mode", ["Standard (Fast)", "Deep (Full Analysis)"], horizontal=True)
    
    if 'opt_queue' not in st.session_state: st.session_state.opt_queue = []
    if 'opt_results' not in st.session_state: st.session_state.opt_results = []
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start_opt = st.button("Start Optimization", type="primary")
    with col_btn2:
        stop_opt = st.button("Stop")

    if start_opt:
        # FULL REQUESTED GRID
        smas = ["MA100", "MA200", "MA300", "MA400", "None"]
        deltas = [0.15, 0.20, 0.30, 0.40, 0.50]
        dtes = [21, 14]
        targets = [75, 50, 25]
        stops = [200, 300, 400, 500]
        
        if scan_depth.startswith("Standard"):
            # A lighter version if the user just wants a quick check
            smas = ["MA200", "None"]
            deltas = [0.20, 0.30]
            dtes = [21, 14]
            targets = [50]
            stops = [200, 300]
        
        grid = list(itertools.product(smas, deltas, dtes, targets, stops))
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
        d_sma, d_delta, d_dte, d_target, d_stop = cfg
        
        remaining = len(st.session_state.opt_queue)
        total_runs = len(st.session_state.opt_results) + remaining + 1
        curr_run = len(st.session_state.opt_results) + 1
        
        st.write(f"**Processing Config {curr_run}/{total_runs}**")
        st.caption(f"Filter: {d_sma} | Delta {d_delta} || Exit: {d_dte} DTE | Target {d_target}% | Stop {d_stop}%")
        
        main_bar = st.progress(curr_run / total_runs)
        
        df_r, bal = run_portfolio_simulation(
            st.session_state.data_map, 
            d_delta, d_sma, 
            d_dte, d_stop, d_target, 
            start_cap, monthly_add, slippage, fees, None
        )
        
        if not df_r.empty:
            wins = df_r[df_r['pnl_dollars'] > 0]['pnl_dollars']
            losses = df_r[df_r['pnl_dollars'] <= 0]['pnl_dollars']
            
            avg_win = wins.mean() if not wins.empty else 0
            avg_loss = losses.mean() if not losses.empty else 0
            
            roi = ((bal - start_cap) / start_cap) * 100
            wr = len(wins) / len(df_r) * 100
            dd_series = (df_r['balance'] - df_r['balance'].cummax()) / df_r['balance'].cummax()
            max_dd = dd_series.min() * 100
            
            st.session_state.opt_results.append({
                "Crash Guard": d_sma,
                "Entry Delta": d_delta, 
                "Exit DTE": d_dte, 
                "Target %": d_target, 
                "Stop %": d_stop,
                "Final Balance": bal, 
                "ROI %": roi, 
                "Win Rate": wr, 
                "Avg Win": avg_win,
                "Avg Loss": avg_loss,
                "Max DD %": max_dd
            })
        st.rerun()

    # RESULTS DISPLAY
    if st.session_state.opt_results and not st.session_state.get('is_running', False):
        res_df = pd.DataFrame(st.session_state.opt_results)
        res_df = res_df.sort_values("Final Balance", ascending=False)
        
        # --- SUMMARY REPORT GENERATOR ---
        best = res_df.iloc[0]
        
        st.divider()
        st.markdown("### 🏆 Strategy Report: The 'Golden' Configuration")
        
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown(f"""
            #### 1. How to FIND Trades
            * **Market Filter:** Only trade when SPY is above the **{best['Crash Guard']}**.
            * **Aggressiveness:** Sell Puts at **{best['Entry Delta']} Delta**.
            """)
        with col_sum2:
            st.markdown(f"""
            #### 2. How to EXIT Trades
            * **Profit Target:** Close early at **{best['Target %']}%** profit.
            * **Defense:** Cut losses if the premium spikes **{best['Stop %']}%**.
            * **Time Limit:** If neither hits, exit at **{best['Exit DTE']} Days** to expiration.
            """)
        
        st.info(f"""
        **Expected Performance:**
        With these settings, the strategy generated **${best['Final Balance']:,.0f}** in equity.
        You can expect to win **{best['Win Rate']:.1f}%** of the time, with an average win of **${best['Avg Win']:.0f}** and an average loss of **${best['Avg Loss']:.0f}**.
        """)
        st.divider()

        st.subheader("Full Leaderboard")
        st.dataframe(res_df.style.format({
            "Final Balance": "${:,.0f}", 
            "ROI %": "{:.1f}%", 
            "Win Rate": "{:.1f}%", 
            "Avg Win": "${:,.0f}",
            "Avg Loss": "${:,.0f}",
            "Max DD %": "{:.1f}%"
        }).background_gradient(subset="Final Balance", cmap="Greens"), use_container_width=True)
