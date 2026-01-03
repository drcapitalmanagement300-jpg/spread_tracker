import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import scipy.stats as si
import time
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
    total_days = len(entry_dates)
    progress_bar = st.progress(0)
    r = 0.045
    
    # 1. GENERATE SIGNALS
    for i, entry_date in enumerate(entry_dates):
        if i % 100 == 0: progress_bar.progress((i + 1) / total_days)

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
            
            # Strike Selection
            best_strike = S_entry * 0.8
            min_delta_diff = 1.0
            scan_strikes = np.linspace(S_entry * 0.70, S_entry, 30)
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
                
                # Percentage of Max Profit Remaining
                # If Credit $1.00, Current Spread $0.50 -> You captured 50%
                profit_captured_pct = ((gross_credit - spread_val) / gross_credit) * 100
                loss_pct = (spread_val / gross_credit) * 100
                
                hit_exit = False
                reason = ""
                
                if days_left < exit_dte_trigger: hit_exit = True; reason = f"DTE < {exit_dte_trigger}"
                elif abs(curr_d_s) > exit_delta_trigger: hit_exit = True; reason = f"Delta > {exit_delta_trigger}"
                elif loss_pct > exit_spread_pct: hit_exit = True; reason = f"Max Loss ({exit_spread_pct}%)"
                # PROFIT TAKER LOGIC
                elif profit_captured_pct >= profit_target_pct: hit_exit = True; reason = f"Profit Target ({profit_target_pct}%)"
                
                if hit_exit:
                    trade_res['exit_reason'] = reason
                    trade_res['pnl'] = gross_credit - spread_val
                    trade_res['exit_date'] = curr_date
                    break
            
            all_potential_trades.append(trade_res)
        except: continue
        
    progress_bar.empty()
    
    # 2. CAPITAL CONSTRAINTS
    if not all_potential_trades: return pd.DataFrame()
    
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
    if not results.empty:
        results['display_entry'] = pd.to_datetime(results['entry_date']).dt.strftime('%Y-%m-%d')
        results['display_exit'] = pd.to_datetime(results['exit_date']).dt.strftime('%Y-%m-%d')
        
    return results, skipped_count

# --- HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""<div style='text-align: left; padding-top: 10px;'><h1 style='margin-bottom: 0px;'>Options Lab</h1><p style='color: gray;'>Portfolio Simulation (Tastytrade Logic)</p></div>""", unsafe_allow_html=True)
with header_col3: st.write("") 
st.markdown("<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: -5px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# --- CONTROLS ---
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    with c1: 
        ticker = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "GLD", "NVDA", "AAPL", "AMD", "TSLA", "MSFT", "AMZN"], index=0)
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker:
            with st.spinner(f"Fetching 10y Data for {ticker}..."):
                st.session_state.df_market = get_market_data(ticker)
                st.session_state.last_ticker = ticker
            
    with c2: exit_delta = st.number_input("Exit Short Delta >", 0.1, 1.0, 0.60, 0.05)
    with c3: exit_spread_pct = st.number_input("Exit Spread Value % >", 100, 1000, 300, 50)
    with c4: exit_dte = st.number_input("Exit DTE <", 0, 30, 21, help="Tastytrade Standard: 21 DTE")
    
    st.markdown("##### ⚙️ Advanced Settings")
    r1, r2, r3, r4 = st.columns(4)
    with r1: start_cap = st.number_input("Capital ($)", 1000, 1000000, 10000, 1000)
    with r2: profit_target = st.number_input("Take Profit %", 10, 90, 50, 5, help="Close early if X% of max profit is reached.")
    with r3: slippage = st.number_input("Slippage ($)", 0.00, 0.10, 0.01, 0.01)
    with r4: fees = st.number_input("Commissions ($)", 0.00, 5.00, 0.00, 0.10)
    
    run_btn = st.button("🔬 Run Simulation", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    if 'df_market' in st.session_state and not st.session_state.df_market.empty:
        with st.spinner(f"Simulating..."):
            results, skipped = run_simulation(
                st.session_state.df_market, 
                0.20, exit_delta, exit_spread_pct, exit_dte, profit_target,
                slippage, fees, start_cap
            )
        
        if results.empty:
            st.warning("No trades generated.")
        else:
            total_trades = len(results)
            wins = results[results['pnl'] > 0]
            losses = results[results['pnl'] <= 0]
            win_rate = len(wins) / total_trades * 100
            total_pnl = results['pnl'].sum() * 100
            
            final_equity = start_cap + total_pnl
            total_return_pct = (total_pnl / start_cap) * 100
            avg_win = wins['pnl'].mean() * 100 if not wins.empty else 0
            avg_loss = losses['pnl'].mean() * 100 if not losses.empty else 0
            
            # --- DISPLAY ---
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Net Profit", f"${total_pnl:,.0f}", delta=f"{total_return_pct:.1f}%")
            m2.metric("Ending Balance", f"${final_equity:,.0f}")
            m3.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W | {len(losses)}L")
            m4.metric("Avg Win", f"${avg_win:.2f}")
            m5.metric("Avg Loss", f"${avg_loss:.2f}")
            
            if skipped > 0:
                st.info(f"ℹ️ **Capital Constraint:** Skipped {skipped} trades due to full account.")
            
            st.markdown("---")
            
            col_chart, col_dist = st.columns([2, 1])
            with col_chart:
                st.subheader("Account Growth")
                results['plot_date'] = pd.to_datetime(results['entry_date'])
                results = results.sort_values("plot_date")
                results['cum_pnl'] = results['pnl'].cumsum() * 100
                results['equity'] = start_cap + results['cum_pnl']
                
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(results['plot_date'], results['equity'], color=SUCCESS_COLOR, linewidth=1.5)
                ax.fill_between(results['plot_date'], results['equity'], start_cap, color=SUCCESS_COLOR, alpha=0.1)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.set_ylabel("Account Value ($)")
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
            
            st.subheader("Trade Log")
            st.dataframe(
                results[['display_entry', 'display_exit', 'short_strike', 'gross_credit', 'exit_reason', 'pnl']],
                use_container_width=True,
                height=300,
                column_config={
                    "display_entry": "Entry",
                    "display_exit": "Exit",
                    "short_strike": "Strike",
                    "gross_credit": "Gross Credit",
                    "exit_reason": "Reason",
                    "pnl": "Net P&L"
                }
            )
    else:
        st.error("Data not loaded. Please select a ticker.")

st.markdown("---")
st.caption("**Model:** Black-Scholes | **Constraints:** Capital Limited + Slippage")
