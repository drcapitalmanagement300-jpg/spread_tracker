import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import google.generativeai as genai
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Finder")

# 1. Initialize Gemini
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        # Fallback or silent error if just testing UI
        pass
except Exception:
    pass

# 2. Universe (Liquid S&P 500 Proxies)
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", 
    "GOOGL", "META", "NFLX", "BAC", "JPM", "XOM", "CVX", "DIS", "BA", "PLTR", "COIN"
]

# --- CUSTOM CSS FOR "CARD" LOOK ---
st.markdown("""
<style>
    /* Metric Label (Small Gray) */
    .metric-label {
        font-size: 12px;
        color: #888;
        margin-bottom: -5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Metric Value (White Bold) */
    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: #FFF;
    }
    /* Price Pill (Red/Green) */
    .price-pill-red {
        background-color: rgba(255, 75, 75, 0.2);
        color: #ff4b4b;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    .price-pill-green {
        background-color: rgba(0, 200, 100, 0.2);
        color: #00c864;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 14px;
    }
    /* Strategy Badge (Yellow) */
    .strategy-badge {
        border: 1px solid #d4ac0d;
        color: #d4ac0d;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
        float: right;
    }
    /* ROC Box (Green Bottom) */
    .roc-box {
        background-color: rgba(0, 255, 127, 0.1);
        border: 1px solid rgba(0, 255, 127, 0.3);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- BLACK-SCHOLES IV CALCULATOR ---
def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    return sigma - fx / vega

def get_implied_volatility(price, strike, time_to_exp, market_price, risk_free_rate=0.045):
    sigma = 0.5 
    try:
        for i in range(50):
            diff = newton_vol_put(price, strike, time_to_exp, market_price, risk_free_rate, sigma)
            sigma = diff
            if abs(diff - sigma) < 1e-5: break
        return abs(sigma)
    except:
        return 0.0

# --- DATA & ANALYSIS ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # HV Rank Calculation
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        rank = 50
        if not hist['HV'].empty:
            mn, mx = hist['HV'].min(), hist['HV'].max()
            if mx != mn: rank = ((hist['HV'].iloc[-1] - mn) / (mx - mn)) * 100

        # Earnings
        earnings_days = 99
        try:
            cal = stock.calendar
            if cal is not None and not cal.empty:
                potential = cal.iloc[0][0] if isinstance(cal.iloc[0][0], (list, tuple)) else cal.iloc[0][0]
                # Simple check if future
                # (Simplified for brevity, assume valid date logic here)
                pass 
        except: pass

        return {
            "price": current_price,
            "change_pct": change_pct,
            "rank": rank,
            "earnings_days": earnings_days,
            "hist": hist
        }
    except: return None

def find_credit_spread(stock_obj, current_price, target_dte=30, width=5.0):
    try:
        exps = stock_obj.options
        if not exps: return None
    except: return None
        
    target_date = datetime.now() + timedelta(days=target_dte)
    best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
    dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
    if dte < 7: return None 

    try:
        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
    except: return None

    # Target Strike ~ 0.96 * Price (Approx 30 Delta)
    target_strike = current_price * 0.96
    short_leg = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]
    if short_leg.empty: return None
    
    short_strike = short_leg.iloc[0]['strike']
    short_mid = (short_leg.iloc[0]['bid'] + short_leg.iloc[0]['ask']) / 2
    
    long_leg = puts.iloc[(puts['strike'] - (short_strike - width)).abs().argsort()[:1]]
    if long_leg.empty: return None
    long_strike = long_leg.iloc[0]['strike']
    
    if abs((short_strike - long_strike) - width) > 1.0: return None

    credit = short_leg.iloc[0]['bid'] - long_leg.iloc[0]['ask']
    max_loss = (short_strike - long_strike) - credit
    
    # Calc IV
    iv = get_implied_volatility(current_price, short_strike, dte/365, short_mid) * 100

    return {
        "expiration": best_exp,
        "dte": dte,
        "short": short_strike,
        "long": long_strike,
        "credit": credit,
        "max_loss": max_loss,
        "iv": iv,
        "roi": (credit / max_loss) * 100 if max_loss > 0 else 0
    }

def get_ai_narrative(ticker, rank):
    # Prompting logic from previous steps
    # Mocking for speed in this visual update
    return {"verdict": "Safe", "explanation": "Sector rotation favoring tech."}

# --- PLOT (Compact) ---
def plot_sparkline_cone(hist, price, iv):
    # Smaller figsize to fit card
    fig, ax = plt.subplots(figsize=(6, 1.5)) 
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Data
    last_90 = hist['Close'].tail(60)
    dates = last_90.index
    days_proj = 30
    
    # Cone
    vol_move = price * (iv/100) * np.sqrt(np.arange(1,31)/365)
    upper = price + vol_move
    lower = price - vol_move
    future_dates = [dates[-1] + timedelta(days=int(i)) for i in range(1,31)]
    
    ax.plot(dates, last_90, color='#00FFAA', lw=1.5)
    ax.fill_between(future_dates, lower, upper, color='#00FFAA', alpha=0.1)
    ax.plot(future_dates, upper, color='gray', linestyle=':', lw=0.5)
    ax.plot(future_dates, lower, color='gray', linestyle=':', lw=0.5)
    
    ax.axis('off') # Turn off axis for clean sparkline look
    return fig

# --- MAIN RENDER LOOP ---

st.title("🦅 Spread Sniper")

with st.sidebar:
    st.header("Filters")
    min_rank = st.slider("Min IV Rank", 0, 100, 30)
    simulate = st.checkbox("Simulate High Vol", True)

if st.button("🔎 Scan Market", type="primary"):
    
    status = st.empty()
    status.write("Scanning...")
    
    # --- SCANNING LOGIC ---
    results = []
    for ticker in LIQUID_TICKERS[:8]: # Limit to 8 for speed demo
        data = get_stock_data(ticker)
        if not data: continue
        
        obj = yf.Ticker(ticker)
        spread = find_credit_spread(obj, data['price'])
        
        if spread:
            rank = data['rank']
            if simulate: rank = np.random.randint(50, 95)
            
            # Simple Scoring
            score = rank + spread['roi']
            
            results.append({
                "ticker": ticker,
                "data": data,
                "spread": spread,
                "score": score
            })
    
    status.empty()
    
    # --- RENDER CARDS ---
    # Sort by Opportunity Score
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    col_layout = st.columns(3) # 3 Cards per row
    
    for i, res in enumerate(results):
        t = res['ticker']
        d = res['data']
        s = res['spread']
        
        # Display in Grid (wraps automatically)
        with col_layout[i % 3]:
            
            # THE CARD CONTAINER
            with st.container(border=True):
                
                # 1. HEADER: Ticker + Pill + Strategy
                # Using HTML to mimic the screenshot layout exactly
                pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 24px; font-weight: 900; color: white;">{t}</span>
                        <span class="{pill_class}">{d['price']:.2f} ({d['change_pct']:.2f}%)</span>
                    </div>
                </div>
                <div style="margin-top: 5px; margin-bottom: 15px;">
                    <span class="strategy-badge">PUT CREDIT SPREAD</span>
                    <div style="clear: both;"></div>
                </div>
                """, unsafe_allow_html=True)

                # 2. DATA GRID (2x2 Layout mostly)
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown(f"""
                    <div class="metric-label">Strikes</div>
                    <div class="metric-value">${s['short']:.0f} / ${s['long']:.0f}</div>
                    <br>
                    <div class="metric-label">Premium</div>
                    <div class="metric-value" style="color:#00FFAA">${s['credit']:.2f}</div>
                    <br>
                    <div class="metric-label">Max Gain</div>
                    <div class="metric-value">${s['credit']*100:.0f}</div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div class="metric-label">DTE</div>
                    <div class="metric-value">{s['dte']} Days</div>
                    <br>
                    <div class="metric-label">Capital Req</div>
                    <div class="metric-value">${s['max_loss']*100:.0f}</div>
                    <br>
                    <div class="metric-label">Max Loss</div>
                    <div class="metric-value" style="color:#FF4B4B">${s['max_loss']*100:.0f}</div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # 3. OUR FUNCTIONALITY (Score + IV)
                kc1, kc2 = st.columns([2, 1])
                with kc1:
                    st.caption("Implied Volatility Cone")
                    st.pyplot(plot_sparkline_cone(d['hist'], d['price'], s['iv']))
                with kc2:
                    st.markdown(f"""
                    <div class="metric-label">IV Rank</div>
                    <div class="metric-value">{d['rank']:.0f}%</div>
                    <br>
                    <div class="metric-label">Opp Score</div>
                    <div class="metric-value" style="color: #d4ac0d">{res['score']:.0f}</div>
                    """, unsafe_allow_html=True)

                # 4. FOOTER (ROC & BUTTON)
                st.markdown(f"""
                <div class="roc-box">
                    <span style="font-size:12px; color: #00c864;">Return on Capital</span><br>
                    <span style="font-size:20px; font-weight:bold; color: #00c864;">{s['roi']:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Add {t}", key=f"add_{t}", use_container_width=True):
                    st.toast(f"Added {t} to Dashboard")
                    if 'portfolio' not in st.session_state: st.session_state.portfolio = []
                    st.session_state.portfolio.append(res)
