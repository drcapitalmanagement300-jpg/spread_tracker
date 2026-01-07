import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from datetime import datetime, timedelta

# Import persistence
from persistence import (
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout 
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Finder")

# --- CONSTANTS & COLORS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
GRID_COLOR = '#444444'
STRIKE_COLOR = '#FF5252'

# --- INITIALIZE DRIVE SERVICE ---
drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# Load trades if logged in
if "trades" not in st.session_state:
    if drive_service:
        st.session_state.trades = load_from_drive(drive_service) or []
    else:
        st.session_state.trades = []

# --- PERSISTENCE: INITIALIZE SCAN RESULTS ---
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

# 1. EXPANDED UNIVERSE
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "SMH", "ARKK", "KRE", "XBI", "GDX",
    "EEM", "FXI", "EWZ", "HYG", "LQD", "UVXY", "BITO", "USO", "UNG", "TQQQ", "SQQQ", "SOXL", "SOXS",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "NFLX", 
    "AVGO", "QCOM", "INTC", "MU", "ARM", "TXN", "AMAT", "LRCX", "ADI", "IBM", "CSCO", "ORCL",
    "PLTR", "CRM", "ADBE", "SNOW", "NOW", "WDAY", "PANW", "CRWD", "DDOG", "NET",
    "COIN", "MSTR", "HOOD", "SQ", "PYPL", "V", "MA", "AFRM", "SOFI",
    "DKNG", "UBER", "ABNB", "ROKU", "SHOP", "DIS", "NKE", "SBUX", "MCD", "WMT", "TGT", "COST", "HD", "LOW", "LULU", "CMG",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK",
    "BA", "CAT", "GE", "F", "GM", "XOM", "CVX", "COP", "OXY", "SLB", "HAL",
    "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "MRNA"
]

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .metric-value { font-size: 16px; font-weight: 700; color: #FFF; }
    .price-pill-red { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .price-pill-green { background-color: rgba(0, 200, 100, 0.15); color: #00c864; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .strategy-badge { border: 1px solid #d4ac0d; color: #d4ac0d; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; }
    .roc-box { background-color: rgba(0, 255, 127, 0.05); border: 1px solid rgba(0, 255, 127, 0.2); border-radius: 6px; padding: 8px; text-align: center; margin-top: 12px; }
    .warning-box { background-color: rgba(211, 47, 47, 0.1); border: 1px solid #d32f2f; border-radius: 8px; padding: 15px; margin-bottom: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- MATH HELPER ---
def get_implied_volatility(price, strike, time_to_exp, market_price, risk_free_rate=0.045):
    try:
        sigma = 0.5
        for i in range(20):
            d1 = (np.log(price / strike) + (risk_free_rate + 0.5 * sigma ** 2) * time_to_exp) / (sigma * np.sqrt(time_to_exp))
            d2 = d1 - sigma * np.sqrt(time_to_exp)
            put_price = strike * np.exp(-risk_free_rate * time_to_exp) * si.norm.cdf(-d2) - price * si.norm.cdf(-d1)
            vega = price * np.sqrt(time_to_exp) * si.norm.pdf(d1)
            diff = put_price - market_price
            if abs(diff) < 1e-5: break
            sigma = sigma - diff / vega
        return abs(sigma)
    except: return 0.0

# --- MARKET HEALTH CHECK ---
def get_market_health():
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y") 
        if hist.empty or len(hist) < 200:
            return True, 0, 0 
        current_price = hist['Close'].iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        is_healthy = current_price > sma_200
        return is_healthy, current_price, sma_200
    except Exception as e:
        return True, 0, 0

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo") 
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = hist['HV'].iloc[-1] if not pd.isna(hist['HV'].iloc[-1]) else 0

        sma_100 = hist['Close'].rolling(window=100).mean().iloc[-1] if len(hist) >= 100 else current_price
        is_above_sma = current_price > sma_100

        rank = 50
        if not hist['HV'].empty:
            mn, mx = hist['HV'].min(), hist['HV'].max()
            if mx != mn: 
                rank = ((current_hv - mn) / (mx - mn)) * 100

        earnings_days = 999
        next_earnings_date_str = "Unknown"
        try:
            future_dates = []
            try:
                dates_df = stock.get_earnings_dates(limit=8)
                if dates_df is not None and not dates_df.empty:
                    for d in dates_df.index:
                        if d.tzinfo: d = d.tz_localize(None)
                        if d.date() > datetime.now().date():
                            future_dates.append(d.date())
            except: pass

            if not future_dates:
                try:
                    cal = stock.calendar
                    if cal is not None:
                        if isinstance(cal, dict):
                            for k in ['Earnings Date', 'Earnings High']:
                                if k in cal:
                                    ds = cal[k]
                                    if isinstance(ds, list):
                                        for d in ds:
                                            if d.date() > datetime.now().date():
                                                future_dates.append(d.date())
                except: pass

            if future_dates:
                next_date = min(future_dates)
                earnings_days = (next_date - datetime.now().date()).days
                next_earnings_date_str = next_date.strftime("%b %d")
        except: pass 

        return {
            "price": current_price,
            "change_pct": change_pct,
            "rank": rank,
            "hv": current_hv,
            "is_above_sma": is_above_sma,
            "earnings_days": earnings_days,
            "earnings_date_str": next_earnings_date_str,
            "hist": hist
        }
    except: return None

# --- TRADE LOGIC (TUNED) ---
def find_optimal_spread(stock_obj, current_price, current_hv, earnings_days, dev_mode=False):
    try:
        exps = stock_obj.options
        if not exps: return None
        
        # 1. DTE TUNING
        # Widen slightly to ensure we catch monthly cycles
        min_days = 14 if dev_mode else 25
        max_days = 60 if dev_mode else 50
        
        target_min_date = datetime.now() + timedelta(days=min_days)
        target_max_date = datetime.now() + timedelta(days=max_days)
        
        valid_exps = []
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d")
                if target_min_date <= edate <= target_max_date:
                    valid_exps.append(e)
            except: pass
            
        if not valid_exps: return None
        best_exp = max(valid_exps) 
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days

        # 2. SAFETY: CLIFF-PROOF
        if not dev_mode and earnings_days <= dte + 1:
            return None

        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        
        atm_puts = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
        imp_vol = atm_puts.iloc[0]['impliedVolatility'] if not atm_puts.empty else (current_hv / 100.0)

        # 3. TARGETING (0.75x EM)
        # Using 0.75x instead of 0.85x to ensure we find candidates in lower vol environments
        expected_move = current_price * imp_vol * np.sqrt(dte/365) * 0.75
        target_short_strike = current_price - expected_move
        
        otm_puts = puts[puts['strike'] <= target_short_strike].sort_values('strike', ascending=False)
        
        # Fallback for Dev Mode
        if otm_puts.empty:
            if dev_mode:
                otm_puts = puts[puts['strike'] < current_price].sort_values('strike', ascending=False)
            else:
                # Try one last check closer to money (standard 1SD)
                target_short_fallback = current_price * 0.95
                otm_puts = puts[puts['strike'] <= target_short_fallback].sort_values('strike', ascending=False)

        if otm_puts.empty: return None

        # 4. WIPEOUT FACTOR (REALISTIC)
        width = 5.0
        # Min Credit: $0.70 (14% Yield)
        min_credit = 0.70 
        if dev_mode: min_credit = 0.05

        best_spread = None

        for index, short_row in otm_puts.iterrows():
            short_strike = short_row['strike']
            bid = short_row['bid']
            ask = short_row['ask']
            
            if ask <= 0: continue
            
            # Liquidity: Tolerant logic
            spread_width = ask - bid
            slippage_pct = spread_width / ask
            is_liquid = dev_mode or (spread_width <= 0.05) or (slippage_pct <= 0.20)
            
            if not is_liquid: continue

            # Long Leg
            long_strike_target = short_strike - width
            long_leg = puts[abs(puts['strike'] - long_strike_target) < 0.2] 
            
            if long_leg.empty: continue 
            long_row = long_leg.iloc[0]
            mid_credit = bid - long_row['ask']
            
            if mid_credit >= min_credit:
                max_loss = width - mid_credit
                roi = (mid_credit / max_loss) * 100 if max_loss > 0 else 0
                iv = short_row['impliedVolatility'] * 100
                
                exp_date_obj = datetime.strptime(best_exp, "%Y-%m-%d")
                exp_date_str = exp_date_obj.strftime("%b %d, %Y")

                best_spread = {
                    "expiration_raw": best_exp,
                    "expiration": exp_date_str,
                    "dte": dte, 
                    "short": short_strike, "long": long_row['strike'],
                    "credit": mid_credit, "max_loss": max_loss, 
                    "iv": iv, "roi": roi,
                    "em": expected_move
                }
                break 
        
        return best_spread

    except: return None

# --- PLOTTING FUNCTION ---
def plot_sparkline_cone(hist, current_price, iv, short_strike, long_strike):
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
    ax.axhline(y=short_strike, color=STRIKE_COLOR, linestyle='-', linewidth=1, alpha=0.9)
    ax.axhline(y=long_strike, color=STRIKE_COLOR, linestyle='-', linewidth=0.8, alpha=0.6)
    ax.fill_between(dates, long_strike, short_strike, color=STRIKE_COLOR, alpha=0.1)

    # Vol Cone
    days_proj = 30
    safe_iv = iv if iv > 0 else 30
    vol_move = current_price * (safe_iv/100) * np.sqrt(np.arange(1, days_proj+1)/365)
    upper_cone = current_price + vol_move
    lower_cone = current_price - vol_move
    future_dates = [dates[-1] + timedelta(days=int(i)) for i in range(1, days_proj+1)]
    
    ax.fill_between(future_dates, lower_cone, upper_cone, color='#00FFAA', alpha=0.1)
    ax.plot(future_dates, upper_cone, color='gray', linestyle=':', lw=0.5)
    ax.plot(future_dates, lower_cone, color='gray', linestyle=':', lw=0.5)
    ax.fill_between(future_dates, long_strike, short_strike, color=STRIKE_COLOR, alpha=0.08)

    ax.grid(True, which='major', linestyle=':', color=GRID_COLOR, alpha=0.3)
    ax.axis('off') 
    plt.tight_layout(pad=0.1)
    return fig

# --- MAIN UI ---
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

# NO DIVIDER LINE HERE - Removed as requested

with st.sidebar:
    st.header("Scanner Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Bypass Filters)", value=False, help="Check this to test in bear markets.")
    if dev_mode: st.warning("⚠️ DEV MODE ACTIVE: Safety Filters Disabled.")

# --- SCAN BUTTON ---
if st.button(f"Scan Market {'(Dev Mode)' if dev_mode else '(Strict)'}"):
    
    # 1. MARKET HEALTH CHECK
    market_healthy, spy_price, spy_sma = get_market_health()
    
    if not market_healthy and not dev_mode:
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="color: #d32f2f; margin-bottom: 0px;">🚫 MARKET CAUTION: BEAR REGIME DETECTED</h3>
            <p style="color: white; font-size: 16px;">SPY is trading at <b>${spy_price:.2f}</b>, below the 200 SMA (<b>${spy_sma:.2f}</b>).</p>
            <p style="color: #bbb; font-size: 14px;">Put Credit Spreads are high-risk in this environment. Enable <b>Dev Mode</b> if you wish to proceed anyway.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # 2. PROCEED IF HEALTHY
    status = st.empty()
    progress = st.progress(0)
    results = []
    
    for i, ticker in enumerate(LIQUID_TICKERS):
        status.caption(f"Analyzing {ticker}...")
        progress.progress((i + 1) / len(LIQUID_TICKERS))
        
        data = get_stock_data(ticker)
        if not data: continue
        
        spread = find_optimal_spread(
            yf.Ticker(ticker), 
            data['price'], 
            data['hv'], 
            data['earnings_days'], 
            dev_mode=dev_mode
        )
        
        if spread:
            # Score Logic
            score = 0
            if spread['iv'] > (data['hv'] + 5.0): score += 40
            elif spread['iv'] > data['hv']: score += 20
            score += 30 # Safety passed
            width = spread['short'] - spread['long']
            if width > 0:
                credit_ratio = spread['credit'] / width
                if credit_ratio >= 0.30: score += 20
                elif credit_ratio >= 0.25: score += 15
                elif credit_ratio >= 0.15: score += 10 # Added tier for acceptable credit
            if data['is_above_sma']: score += 10
            
            display_score = min(score, 100.0)
            
            # Lowered display threshold slightly to catch 'Good' trades
            if dev_mode or display_score >= 50:
                results.append({"ticker": ticker, "data": data, "spread": spread, "score": score, "display_score": display_score})
    
    progress.empty()
    status.empty()
    st.session_state.scan_results = sorted(results, key=lambda x: x['score'], reverse=True)

# --- DISPLAY LOGIC ---
if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    if not results:
        st.info("No setups found meeting criteria.")
    else:
        st.success(f"Found {len(results)} High-Probability Opportunities")
        cols = st.columns(3)
        for i, res in enumerate(results):
            t = res['ticker']
            d = res['data']
            s = res['spread']
            
            with cols[i % 3]:
                with st.container(border=True):
                    pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                    badge_text = "ELITE EDGE" if res['display_score'] >= 80 else "SOLID SETUP"
                    badge_style = "border: 1px solid #00C853; color: #00C853;" if res['display_score'] >= 80 else "border: 1px solid #d4ac0d; color: #d4ac0d;"
                    
                    if dev_mode and res['display_score'] < 50:
                        badge_text = "TEST RESULT"
                        badge_style = "border: 1px solid gray; color: gray;"

                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-size: 22px; font-weight: 900; color: white; line-height: 1;">{t}</div>
                            <div style="margin-top: 4px;"><span class="{pill_class}">${d['price']:.2f} ({d['change_pct']:.2f}%)</span></div>
                        </div>
                        <div class="strategy-badge" style="{badge_style}">{badge_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.divider()
                    
                    # Calculate Exit Target (21 Days prior to Exp)
                    exp_dt = datetime.strptime(s['expiration_raw'], "%Y-%m-%d")
                    exit_dt = exp_dt - timedelta(days=21)
                    exit_str = exit_dt.strftime("%b %d, %Y")

                    c1, c2 = st.columns(2)
                    with c1:
                        # Added Max Risk side-by-side with Credit
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
                        <div style="font-size: 10px; color: gray;">{s['expiration']}</div>
                        <div style="font-size: 10px; color: #FFA726; margin-top: 2px;">Exit: {exit_str}</div>
                        <div style="height: 8px;"></div>
                        <div class="metric-label">Implied Vol</div>
                        <div class="metric-value">{s['iv']:.1f}%</div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    vc1, vc2 = st.columns([2, 1])
                    with vc1: st.pyplot(plot_sparkline_cone(d['hist'], d['price'], s['iv'], s['short'], s['long']), use_container_width=True)
                    with vc2:
                        st.markdown(f"""
                        <div class="metric-label" style="text-align: right;">Edge Score</div>
                        <div class="metric-value" style="text-align: right; color: #d4ac0d;">{res['display_score']:.0f}</div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""<div class="roc-box"><span style="font-size:11px; color: #00c864; text-transform: uppercase;">Return on Capital</span><br><span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.2f}%</span></div>""", unsafe_allow_html=True)
                    
                    add_key = f"add_mode_{t}_{i}"
                    st.write("") 
                    if st.button(f"Add {t}", key=f"btn_{t}_{i}", use_container_width=True):
                        st.session_state[add_key] = True

                    if st.session_state.get(add_key, False):
                        st.markdown("##### Size")
                        num = st.number_input(f"Contracts", min_value=1, value=1, key=f"c_{t}_{i}")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            if st.button("✅", key=f"ok_{t}_{i}"):
                                new_trade = {
                                    "id": f"{t}-{s['short']}-{s['expiration_raw']}",
                                    "ticker": t, "contracts": num, 
                                    "short_strike": s['short'], "long_strike": s['long'],
                                    "expiration": s['expiration_raw'], "credit": s['credit'],
                                    "entry_date": datetime.now().date().isoformat(),
                                    "pnl_history": []
                                }
                                st.session_state.trades.append(new_trade)
                                if drive_service: save_to_drive(drive_service, st.session_state.trades)
                                st.toast(f"Added {t}")
                                del st.session_state[add_key]
                                st.rerun()
                        with cc2:
                            if st.button("❌", key=f"no_{t}_{i}"):
                                del st.session_state[add_key]
                                st.rerun()
