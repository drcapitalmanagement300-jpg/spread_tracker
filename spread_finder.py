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
st.set_page_config(layout="wide", page_title="Spread Sniper Pro")

# --- CONSTANTS & COLORS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
GRID_COLOR = '#444444'
STRIKE_COLOR = '#FF5252'
WHITE_DIVIDER_HTML = "<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>"

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

# 1. UNIVERSE (~60 Tickers)
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE", "SMH", "ARKK",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "NFLX", 
    "AVGO", "QCOM", "INTC", "MU", "ARM", "PLTR", "CRM", "ADBE",
    "COIN", "MSTR", "HOOD", "DKNG", "UBER", "ABNB", "SQ", "ROKU", "SHOP",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "PYPL",
    "DIS", "NKE", "SBUX", "MCD", "WMT", "TGT", "COST", "HD", "LOW",
    "BA", "CAT", "GE", "F", "GM", "XOM", "CVX", "COP"
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
    .earnings-box { background-color: rgba(255, 255, 255, 0.05); border: 1px solid #444; border-radius: 6px; padding: 8px; text-align: center; margin-top: 10px; font-size: 13px; color: #E8EAED; }
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

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        
        rank = 50
        if not hist['HV'].empty:
            mn, mx = hist['HV'].min(), hist['HV'].max()
            if mx != mn: 
                rank = ((hist['HV'].iloc[-1] - mn) / (mx - mn)) * 100

        # --- EARNINGS LOGIC ---
        earnings_days = 999
        next_earnings_date_str = "N/A"
        try:
            # Try getting calendar
            cal = stock.calendar
            future_dates = []
            
            # Handle different yfinance versions/return types
            if cal is not None and not cal.empty:
                if isinstance(cal, pd.DataFrame):
                    # Transpose if needed or grab the first row/column
                    # Often cal.iloc[0] contains the dates
                    dates = cal.iloc[0].values if not cal.iloc[0].empty else cal.index
                else:
                    dates = cal.values() # If dict
                
                for d in dates:
                    # Convert to datetime object if needed
                    if isinstance(d, (np.datetime64, pd.Timestamp, datetime)):
                        d_date = pd.to_datetime(d).date()
                        if d_date > datetime.now().date():
                            future_dates.append(d_date)
                
                if future_dates:
                    next_date = min(future_dates)
                    earnings_days = (next_date - datetime.now().date()).days
                    next_earnings_date_str = next_date.strftime("%b %d, %Y")
        except: 
            # Fallback if calendar fails
            next_earnings_date_str = "Unknown"

        return {
            "price": current_price,
            "change_pct": change_pct,
            "rank": rank,
            "earnings_days": earnings_days,
            "earnings_date_str": next_earnings_date_str,
            "hist": hist
        }
    except: return None

# --- TRADE LOGIC ---
def find_optimal_spread(stock_obj, current_price, target_dte=30, dev_mode=False):
    try:
        exps = stock_obj.options
        if not exps: return None
        
        target_date = datetime.now() + timedelta(days=target_dte)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
        
        if dte < 20 or dte > 50: return None 

        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        
        target_strike = current_price * 0.95 
        short_leg = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]
        if short_leg.empty: return None
        
        short_strike = short_leg.iloc[0]['strike']
        short_bid = short_leg.iloc[0]['bid']
        short_ask = short_leg.iloc[0]['ask']
        
        if short_ask == 0: return None 
        if not dev_mode and (short_ask - short_bid) > 0.50: return None

        long_strike_target = short_strike - 5.0
        long_leg = puts.iloc[(puts['strike'] - long_strike_target).abs().argsort()[:1]]
        if long_leg.empty: return None
        
        long_strike = long_leg.iloc[0]['strike']
        long_ask = long_leg.iloc[0]['ask']
        
        if abs((short_strike - long_strike) - 5.0) > 1.0: return None

        credit = short_bid - long_ask 
        max_loss = (short_strike - long_strike) - credit
        
        # In Dev Mode, accept any credit > 0
        min_credit = 0.05 if dev_mode else 0.40
        if credit < min_credit: return None

        roi = (credit / max_loss) * 100 if max_loss > 0 else 0
        iv = get_implied_volatility(current_price, short_strike, dte/365, (short_bid+short_ask)/2) * 100

        return {
            "expiration": best_exp, "dte": dte, 
            "short": short_strike, "long": long_strike,
            "credit": credit, "max_loss": max_loss, 
            "iv": iv, "roi": roi
        }
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

# ---------------- Header ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])

with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except Exception:
        st.write("**DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Spread Finder</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.write("") 

st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Scanner Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Bypass Filters)", value=False, help="Check this to see ALL valid spread structures.")
    
    if dev_mode:
        st.warning("⚠️ DEV MODE ACTIVE: Filters are disabled.")

# --- SCAN BUTTON LOGIC ---
if st.button(f"Scan Market {'(Dev Mode)' if dev_mode else '(Strict)'}"):
    
    status = st.empty()
    progress = st.progress(0)
    results = []
    
    for i, ticker in enumerate(LIQUID_TICKERS):
        status.caption(f"Analyzing {ticker}...")
        progress.progress((i + 1) / len(LIQUID_TICKERS))
        
        # 1. Fetch
        data = get_stock_data(ticker)
        if not data: continue
        
        # 2. Filter Rank
        if not dev_mode and data['rank'] < 80: continue 

        # 3. Find Trade
        obj = yf.Ticker(ticker)
        spread = find_optimal_spread(obj, data['price'], dev_mode=dev_mode)
        
        if spread:
            # 4. Filter Earnings
            if not dev_mode and data['earnings_days'] < spread['dte'] + 2:
                continue 

            score = data['rank'] + spread['roi']
            results.append({"ticker": ticker, "data": data, "spread": spread, "score": score})
    
    progress.empty()
    status.empty()
    
    st.session_state.scan_results = sorted(results, key=lambda x: x['score'], reverse=True)

# --- DISPLAY LOGIC ---
if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    
    if not results:
        st.info("No setups found.")
    else:
        st.success(f"Found {len(results)} Opportunities")
        
        cols = st.columns(3)
        for i, res in enumerate(results):
            t = res['ticker']
            d = res['data']
            s = res['spread']
            
            with cols[i % 3]:
                with st.container(border=True):
                    pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                    
                    badge_text = "PRIME SETUP"
                    badge_style = "border: 1px solid #d4ac0d; color: #d4ac0d;" 
                    if dev_mode and (d['rank'] < 80 or s['credit'] < 0.40):
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

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-label">Strikes</div>
                        <div class="metric-value">${s['short']:.0f} / ${s['long']:.0f}</div>
                        <div style="height: 8px;"></div>
                        <div class="metric-label">Credit</div>
                        <div class="metric-value" style="color:{SUCCESS_COLOR}">${s['credit']:.2f}</div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-label">Expiry</div>
                        <div class="metric-value">{s['dte']} Days</div>
                        <div style="height: 8px;"></div>
                        <div class="metric-label">Max Risk</div>
                        <div class="metric-value" style="color:#FF4B4B">${s['max_loss']*100:.0f}</div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")
                    
                    vc1, vc2 = st.columns([2, 1])
                    with vc1:
                         st.pyplot(plot_sparkline_cone(d['hist'], d['price'], s['iv'], s['short'], s['long']), use_container_width=True)
                    with vc2:
                        st.markdown(f"""
                        <div class="metric-label" style="text-align: right;">IV Rank</div>
                        <div class="metric-value" style="text-align: right;">{d['rank']:.0f}%</div>
                        <div style="height: 10px;"></div>
                        <div class="metric-label" style="text-align: right;">Opp Score</div>
                        <div class="metric-value" style="text-align: right; color: #d4ac0d;">{res['score']:.0f}</div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="roc-box">
                        <span style="font-size:11px; color: #00c864; text-transform: uppercase; letter-spacing: 1px;">Return on Capital</span><br>
                        <span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- EARNINGS DISPLAY (Replaces AI) ---
                    # Show the actual date or N/A
                    earnings_label = d['earnings_date_str'] if d['earnings_date_str'] != "Unknown" else "Unknown"
                    st.markdown(f"""
                    <div class="earnings-box">
                        <span style="color: #AAA;">Next Earnings:</span> <strong style="color: white;">{earnings_label}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- ADD TO DASHBOARD LOGIC ---
                    add_key = f"add_mode_{t}_{i}"
                    
                    st.write("") 
                    
                    if st.button(f"Add {t} to Dashboard", key=f"btn_{t}_{i}", use_container_width=True):
                        if not drive_service:
                            st.error("Please sign in with Google on the Dashboard page first.")
                        else:
                            st.session_state[add_key] = True

                    if st.session_state.get(add_key, False):
                        st.markdown("##### Position Sizing")
                        num_contracts = st.number_input(f"Contracts for {t}", min_value=1, value=1, step=1, key=f"contracts_{t}_{i}")
                        
                        col_conf, col_can = st.columns(2)
                        with col_conf:
                            if st.button("✅ Confirm", key=f"conf_{t}_{i}"):
                                new_trade = {
                                    "id": f"{t}-{s['short']}-{s['long']}-{s['expiration']}",
                                    "ticker": t,
                                    "contracts": num_contracts, 
                                    "short_strike": s['short'],
                                    "long_strike": s['long'],
                                    "expiration": s['expiration'],
                                    "credit": s['credit'],
                                    "entry_date": datetime.now().date().isoformat(),
                                    "created_at": datetime.utcnow().isoformat(),
                                    "cached": {},
                                    "pnl_history": []
                                }
                                
                                st.session_state.trades.append(new_trade)
                                
                                if drive_service:
                                    save_to_drive(drive_service, st.session_state.trades)
                                    st.success(f"Success: Added {num_contracts}x {t} to Dashboard!")
                                    st.toast(f"Successfully added {num_contracts}x {t} to Dashboard!")
                                
                                del st.session_state[add_key]
                                st.rerun()
                                
                        with col_can:
                            if st.button("❌ Cancel", key=f"canc_{t}_{i}"):
                                del st.session_state[add_key]
                                st.rerun()
