import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Finder (Local)")

# --- PERSISTENCE (GOOGLE DRIVE) ---
try:
    from persistence import (
        build_drive_service_from_session,
        save_to_drive,
        load_from_drive
    )
except ImportError:
    st.error("Missing 'persistence.py'. Please copy it from your GitHub repo to this folder.")
    st.stop()

# --- INIT STATE ---
if "init_done" not in st.session_state:
    st.session_state.init_done = True
    st.session_state.scan_results = []
    st.session_state.scan_log = []
    
    # Connect to Drive immediately
    try:
        drive_service = build_drive_service_from_session()
        st.session_state.trades = load_from_drive(drive_service) or []
    except Exception as e:
        st.error(f"Drive Error: {e}")
        st.session_state.trades = []

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
GRID_COLOR = '#444444'
STRIKE_COLOR = '#FF5252'

# Top Liquid Tickers for Options
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE", 
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "NFLX", 
    "AVGO", "QCOM", "INTC", "MU", "ARM", "TXN", "AMAT", "LRCX", "ADI", 
    "PLTR", "COIN", "MSTR", "HOOD", "SQ", "PYPL", "V", "MA", "UBER", 
    "SHOP", "DIS", "BA", "CAT", "GE", "F", "GM", "XOM", "CVX", "LLY", "UNH", "JNJ"
]

ETFS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE"]

# --- HELPER FUNCTIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def process_bulk_data(df, ticker):
    try:
        hist = pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try: ticker_df = df[ticker].copy()
            except: return None
        else: ticker_df = df.copy()

        if 'Close' not in ticker_df.columns: return None
        hist['Close'] = ticker_df['Close']
        hist['Open'] = ticker_df['Open'] if 'Open' in ticker_df.columns else hist['Close']
        hist = hist.dropna()
        if hist.empty: return None

        current_price = hist['Close'].iloc[-1]
        
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = hist['HV'].iloc[-1] if not pd.isna(hist['HV'].iloc[-1]) else 0

        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        # Bollinger Bands
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['STD_20'] = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['SMA_20'] + (hist['STD_20'] * 2)
        hist['BB_Lower'] = hist['SMA_20'] - (hist['STD_20'] * 2)

        sma_200 = hist['SMA_200'].iloc[-1] if len(hist) >= 200 else current_price
        bb_upper = hist['BB_Upper'].iloc[-1] if not pd.isna(hist['BB_Upper'].iloc[-1]) else current_price * 1.05
        bb_lower = hist['BB_Lower'].iloc[-1] if not pd.isna(hist['BB_Lower'].iloc[-1]) else current_price * 0.95
        current_rsi = hist['RSI'].iloc[-1] if len(hist) >= 14 else 50

        is_uptrend = current_price > sma_200
        is_oversold_bb = current_price <= (bb_lower * 1.01)
        is_overbought_bb = current_price >= (bb_upper * 0.99)
        
        change_pct = 0.0
        if len(hist) > 1:
            prev = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev) / prev) * 100

        type_str = "(ETF)" if ticker in ETFS else "(Stock)"

        return {
            "price": current_price, "change_pct": change_pct, "hv": current_hv,
            "is_uptrend": is_uptrend, "rsi": current_rsi,
            "is_oversold_bb": is_oversold_bb, "is_overbought_bb": is_overbought_bb, 
            "hist": hist, "type_str": type_str
        }
    except: return None

def find_optimal_spread(ticker, stock_obj, current_price, current_hv, spread_width_target=5.0, dev_mode=False):
    try:
        # Standard yfinance fetch - works reliably on home internet
        exps = stock_obj.options
        if not exps: return None, "No Options"
        
        min_days = 14 if dev_mode else 25
        max_days = 60 if dev_mode else 50
        
        valid_exps = []
        now = datetime.now()
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d")
                if (now + timedelta(days=min_days)) <= edate <= (now + timedelta(days=max_days)):
                    valid_exps.append(e)
            except: pass
            
        if not valid_exps: return None, "No Valid Expiry"
        best_exp = max(valid_exps) 
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - now).days

        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        
        if puts.empty: return None, "No Puts"

        atm_puts = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
        imp_vol = atm_puts.iloc[0]['impliedVolatility'] if not atm_puts.empty else (current_hv / 100.0)

        # Expected Move (0.75 SD)
        expected_move = current_price * imp_vol * np.sqrt(dte/365) * 0.75
        target_short_strike = current_price - expected_move
        
        otm_puts = puts[puts['strike'] <= target_short_strike].sort_values('strike', ascending=False)
        
        if otm_puts.empty:
            if dev_mode: otm_puts = puts[puts['strike'] < current_price].sort_values('strike', ascending=False)
            else:
                target_fallback = current_price * 0.95
                otm_puts = puts[puts['strike'] <= target_fallback].sort_values('strike', ascending=False)
        
        if otm_puts.empty: return None, "No Safe Strikes"

        width = float(spread_width_target)
        min_credit = 0.50 if ticker in ETFS else 0.70
        if dev_mode: min_credit = 0.05

        best_spread = None
        rejection_reason = f"Credit < ${min_credit}"

        for index, short_row in otm_puts.iterrows():
            short_strike = short_row['strike']
            bid = short_row['bid']
            ask = short_row['ask']
            if ask <= 0: continue
            
            # Liquidity
            spread_width = ask - bid
            slippage_pct = spread_width / ask
            max_spread_allowed = 0.20 if ticker in ETFS else 0.50
            
            is_liquid = dev_mode or (spread_width <= max_spread_allowed) or (slippage_pct <= 0.25)
            if not is_liquid: 
                rejection_reason = f"Illiquid ({spread_width:.2f})"
                continue

            long_target = short_strike - width
            long_leg = puts[abs(puts['strike'] - long_target) < 0.5] 
            if long_leg.empty: continue 
            long_row = long_leg.iloc[0]
            mid_credit = bid - long_row['ask']
            
            if mid_credit >= min_credit:
                max_loss = width - mid_credit
                roi = (mid_credit / max_loss) * 100 if max_loss > 0 else 0
                iv = short_row['impliedVolatility'] * 100
                
                exp_str = datetime.strptime(best_exp, "%Y-%m-%d").strftime("%b %d, %Y")

                best_spread = {
                    "expiration_raw": best_exp, "expiration": exp_str, "dte": dte, 
                    "short": short_strike, "long": long_row['strike'],
                    "credit": mid_credit, "max_loss": max_loss, 
                    "iv": iv, "roi": roi, "em": expected_move
                }
                break 
        
        return (best_spread, None) if best_spread else (None, rejection_reason)

    except Exception as e: return None, f"Error: {str(e)}"

# --- PLOTTING ---
def plot_sparkline_cone(hist, short_strike, long_strike, current_price, iv, dte):
    fig, ax = plt.subplots(figsize=(4, 1.3)) 
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    last_60 = hist.tail(60).copy()
    dates = last_60.index
    if 'Open' not in last_60.columns: last_60['Open'] = last_60['Close']
    bar_colors = np.where(last_60['Close'] >= last_60['Open'], SUCCESS_COLOR, WARNING_COLOR)
    heights = last_60['Close'] - last_60['Open']
    bottoms = last_60['Open']
    
    ax.bar(dates, heights, bottom=bottoms, color=bar_colors, width=0.8, align='center', alpha=0.9)
    
    if iv > 0 and dte > 0:
        last_date = dates[-1]
        future_days = np.arange(1, dte + 5)
        future_dates = [last_date + timedelta(days=int(d)) for d in future_days]
        std_move = current_price * (iv / 100.0) * np.sqrt(future_days / 365.0)
        upper_cone = current_price + std_move
        lower_cone = current_price - std_move
        
        # GREEN CONE
        ax.plot(future_dates, upper_cone, color=SUCCESS_COLOR, linestyle=':', linewidth=1, alpha=0.6)
        ax.plot(future_dates, lower_cone, color=SUCCESS_COLOR, linestyle=':', linewidth=1, alpha=0.6)
        ax.fill_between(future_dates, lower_cone, upper_cone, color=SUCCESS_COLOR, alpha=0.1)

    ax.axhline(y=short_strike, color=STRIKE_COLOR, linestyle='-', linewidth=1, alpha=0.9)
    ax.axhline(y=long_strike, color=STRIKE_COLOR, linestyle='-', linewidth=0.8, alpha=0.6)
    
    full_dates = list(dates) 
    if iv > 0: full_dates.extend(future_dates)
    ax.fill_between(full_dates, long_strike, short_strike, color=STRIKE_COLOR, alpha=0.1)

    ax.grid(True, which='major', linestyle=':', color=GRID_COLOR, alpha=0.3)
    ax.axis('off') 
    plt.tight_layout(pad=0.1)
    return fig

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .metric-value { font-size: 16px; font-weight: 700; color: #FFF; }
    .price-pill-red { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .price-pill-green { background-color: rgba(0, 200, 100, 0.15); color: #00c864; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    
    .strategy-badge { 
        color: #d4ac0d; 
        padding: 2px 8px; 
        font-size: 12px; 
        font-weight: bold; 
        letter-spacing: 1px; 
        text-transform: uppercase; 
        text-align: right;
    }
    
    .roc-box { background-color: rgba(0, 255, 127, 0.05); border: 1px solid rgba(0, 255, 127, 0.2); border-radius: 6px; padding: 8px; text-align: center; margin-top: 12px; }
    .stCodeBlock { font-family: 'Courier New', monospace; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# --- PROFESSIONAL HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Spread Finder</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Scanner Settings")
    dev_mode = st.checkbox("Dev Mode (Bypass Filters)", value=False)
    width_target = st.slider("Width ($)", 1, 25, 5, 1)
    if st.button("Reset Scanner"):
        st.session_state.scan_results = []
        st.session_state.scan_log = []
        st.session_state.batch_complete = False
        st.rerun()

# --- SCANNER ---
if st.button("Scan Market"):
    st.session_state.scan_results = []
    st.session_state.scan_log = []
    st.session_state.scan_complete = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    live_log_container = st.empty()
    
    # Check drive
    if not drive_service:
        st.error("Drive Connection Failed. Check .streamlit/secrets.toml")
        st.stop()
    
    batch_size = 5
    total_tickers = len(LIQUID_TICKERS)
    
    for start_idx in range(0, total_tickers, batch_size):
        end_idx = min(start_idx + batch_size, total_tickers)
        batch = LIQUID_TICKERS[start_idx:end_idx]
        
        status_text.text(f"Scanning Batch {start_idx}-{end_idx} of {total_tickers}...")
        
        try:
            bulk_data = yf.download(batch, period="1y", group_by='ticker', progress=False)
            st.session_state.scan_log.append(f"[BATCH] Data for {len(batch)} tickers")
        except:
            bulk_data = None
            
        for i, ticker in enumerate(batch):
            progress_val = (start_idx + i) / total_tickers
            progress_bar.progress(progress_val)
            
            if bulk_data is not None and not bulk_data.empty:
                data = process_bulk_data(bulk_data, ticker)
            else:
                data = None
                
            if not data:
                st.session_state.scan_log.append(f"[{ticker}] No Data")
                continue
            
            time.sleep(0.5)
            
            spread, reject_reason = find_optimal_spread(
                ticker, 
                yf.Ticker(ticker), 
                data['price'], 
                data['hv'], 
                spread_width_target=width_target, 
                dev_mode=dev_mode
            )
            
            if spread:
                st.session_state.scan_log.append(f"[{ticker}] ✅ FOUND: Credit ${spread['credit']:.2f}")
                score = 0
                if spread['iv'] > (data['hv'] + 5.0): score += 30
                elif spread['iv'] > data['hv']: score += 15
                if data['is_uptrend']: score += 10
                if data['is_uptrend'] and data['is_oversold_bb']: score += 30
                
                score += 20 
                display_score = min(score, 100.0)
                
                if dev_mode or display_score >= 50:
                    st.session_state.scan_results.append({
                        "ticker": ticker, "data": data, "spread": spread, 
                        "score": score, "display_score": display_score
                    })
            else:
                st.session_state.scan_log.append(f"[{ticker}] ⏭️ {reject_reason}")
            
            log_text = "\n".join(st.session_state.scan_log[-12:]) 
            live_log_container.code(log_text, language="text")
            
    st.session_state.scan_complete = True
    progress_bar.empty()
    status_text.empty()
    live_log_container.empty() 

# --- DISPLAY RESULTS ---
if st.session_state.get('scan_complete', False) and st.session_state.scan_results:
    sorted_results = sorted(st.session_state.scan_results, key=lambda x: x['score'], reverse=True)
    st.success(f"Scan Complete: Found {len(sorted_results)} Opportunities")
    
    cols = st.columns(3)
    for i, res in enumerate(sorted_results):
        t = res['ticker']
        d = res['data']
        s = res['spread']
        
        with cols[i % 3]:
            with st.container(border=True):
                pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                badge_text = "ELITE EDGE" if res['display_score'] >= 80 else "SOLID SETUP"
                
                if d['is_uptrend'] and (d['is_oversold_bb'] or d['rsi'] < 45):
                    signal_html = "<span style='color: #00FFAA; font-weight: bold; font-size: 14px;'>BUY NOW (DIP)</span>"
                elif d['is_uptrend']:
                    signal_html = "<span style='color: #FFA726; font-weight: bold; font-size: 14px;'>WAIT (NEUTRAL)</span>"
                else:
                    signal_html = "<span style='color: #FF5252; font-weight: bold; font-size: 14px;'>PASS (TREND)</span>"

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 22px; font-weight: 900; color: white; line-height: 1;">
                            {t} <span style="font-size: 12px; font-weight: 400; color: #aaa;">{d['type_str']}</span>
                        </div>
                        <div style="margin-top: 2px;"><span class="{pill_class}">${d['price']:.2f} ({d['change_pct']:.2f}%)</span></div>
                    </div>
                    <div style="text-align: right;">
                        <div class="strategy-badge">{badge_text}</div>
                        <div style="font-size: 20px; font-weight: 900; color: #d4ac0d; margin-top: 4px;">{res['display_score']:.0f}</div>
                        <div style="font-size: 9px; color: #666; text-transform: uppercase;">Edge Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-label">Strikes</div>
                    <div class="metric-value">${s['short']:.0f} / ${s['long']:.0f}</div>
                    <div style="height: 8px;"></div>
                    <div style="display: flex; gap: 15px;">
                        <div>
                            <div class="metric-label">Credit</div>
                            <div class="metric-value" style="color:{SUCCESS_COLOR}">${s['credit']:.2f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Max Risk</div>
                            <div class="metric-value" style="color:#FF5252">${s['max_loss']:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-label">Expiry</div>
                    <div class="metric-value">{s['dte']} Days</div>
                    <div style="height: 8px;"></div>
                    <div class="metric-label">Action Signal</div>
                    <div>{signal_html}</div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.pyplot(plot_sparkline_cone(d['hist'], s['short'], s['long'], d['price'], s['iv'], s['dte']), use_container_width=True)
                st.markdown(f"""<div class="roc-box"><span style="font-size:11px; color: #00c864; text-transform: uppercase;">Return on Capital</span><br><span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.2f}%</span></div>""", unsafe_allow_html=True)
                
                add_key = f"add_mode_{t}_{i}"
                st.write("") 
                if st.button(f"Add Trade", key=f"btn_{t}_{i}", use_container_width=True):
                    st.session_state[add_key] = True

                if st.session_state.get(add_key, False):
                    st.markdown("##### Size")
                    num = st.number_input(f"Contracts", min_value=1, value=1, key=f"c_{t}_{i}")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Confirm Add", key=f"ok_{t}_{i}"):
                            new_trade = {
                                "id": f"{t}-{s['short']}-{s['expiration_raw']}",
                                "ticker": t, "contracts": num, 
                                "short_strike": s['short'], "long_strike": s['long'],
                                "expiration": s['expiration_raw'], "credit": s['credit'],
                                "entry_date": datetime.now().date().isoformat(),
                                "pnl_history": []
                            }
                            st.session_state.trades.append(new_trade)
                            
                            # --- SAVE TO CLOUD ---
                            if drive_service: save_to_drive(drive_service, st.session_state.trades)
                            
                            st.toast(f"Added {t} to Cloud Dashboard")
                            del st.session_state[add_key]
                            st.rerun()
                    with cc2:
                        if st.button("Cancel", key=f"no_{t}_{i}"):
                            del st.session_state[add_key]
                            st.rerun()

# --- BOTTOM LOG ---
if st.session_state.scan_log:
    with st.expander("Show Full Scanner Log", expanded=True):
        st.code("\n".join(st.session_state.scan_log), language="text")
