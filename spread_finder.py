import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import google.generativeai as genai
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Sniper")

# 1. Initialize Gemini
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception:
    pass

# 2. UNIVERSE: EXPANDED LIQUIDITY LIST (~60 Tickers)
# Includes Top Holdings of S&P 500, Nasdaq 100, and High-Beta Momentum Stocks
LIQUID_TICKERS = [
    # INDICES & ETFS
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE", "SMH", "ARKK",
    # MAGNIFICENT 7 + TECH
    "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", "GOOGL", "NFLX", 
    "AVGO", "QCOM", "INTC", "MU", "ARM", "PLTR", "CRM", "ADBE",
    # HIGH BETA / CRYPTO PROXIES
    "COIN", "MSTR", "HOOD", "DKNG", "UBER", "ABNB", "SQ", "ROKU", "SHOP",
    # FINANCIALS
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "PYPL",
    # RETAIL & CONSUMER
    "DIS", "NKE", "SBUX", "MCD", "WMT", "TGT", "COST", "HD", "LOW",
    # INDUSTRIAL & ENERGY
    "BA", "CAT", "GE", "F", "GM", "XOM", "CVX", "COP"
]

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Metric Typography */
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .metric-value { font-size: 16px; font-weight: 700; color: #FFF; }
    
    /* Pills & Badges */
    .price-pill-red { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .price-pill-green { background-color: rgba(0, 200, 100, 0.15); color: #00c864; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .strategy-badge { border: 1px solid #d4ac0d; color: #d4ac0d; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; }
    
    /* Footer Box */
    .roc-box { background-color: rgba(0, 255, 127, 0.05); border: 1px solid rgba(0, 255, 127, 0.2); border-radius: 6px; padding: 8px; text-align: center; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)


# --- MATH & CALCULATORS ---

def newton_vol_put(S, K, T, P, r, sigma):
    """Back-solves IV for a Put option."""
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
        vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
        return sigma - fx / vega
    except:
        return sigma

def get_implied_volatility(price, strike, time_to_exp, market_price, risk_free_rate=0.045):
    """Wrapper for Newton-Raphson solver."""
    sigma = 0.5 
    try:
        for i in range(50):
            diff = newton_vol_put(price, strike, time_to_exp, market_price, risk_free_rate, sigma)
            sigma = diff
            if abs(diff - sigma) < 1e-5: break
        return abs(sigma)
    except: return 0.0

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """Fetches history and calculates Volatility Rank (Proxy)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # --- PROXY: REALIZED VOLATILITY RANK ---
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        
        rank = 50
        if not hist['HV'].empty:
            mn, mx = hist['HV'].min(), hist['HV'].max()
            if mx != mn: 
                rank = ((hist['HV'].iloc[-1] - mn) / (mx - mn)) * 100

        result = {
            "price": current_price,
            "change_pct": change_pct,
            "rank": rank,
            "hist": hist
        }
        return result
    except: return None

def find_credit_spread(stock_obj, current_price, target_dte=30, width=5.0):
    """Finds the 30-Delta Put Credit Spread."""
    try:
        exps = stock_obj.options
        if not exps: return None
        target_date = datetime.now() + timedelta(days=target_dte)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
        if dte < 7: return None 

        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        
        # Target 30 Delta
        target_strike = current_price * 0.96
        short_leg = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]
        if short_leg.empty: return None
        
        short_strike = short_leg.iloc[0]['strike']
        short_mid = (short_leg.iloc[0]['bid'] + short_leg.iloc[0]['ask']) / 2
        
        long_leg = puts.iloc[(puts['strike'] - (short_strike - width)).abs().argsort()[:1]]
        if long_leg.empty: return None
        long_strike = long_leg.iloc[0]['strike']
        
        # Strict Width Check (Allowing small variance for weird strikes)
        if abs((short_strike - long_strike) - width) > 1.0: return None

        credit = short_leg.iloc[0]['bid'] - long_leg.iloc[0]['ask']
        max_loss = (short_strike - long_strike) - credit
        
        # Calculate Spread IV (Live)
        iv = get_implied_volatility(current_price, short_strike, dte/365, short_mid) * 100
        roi = (credit / max_loss) * 100 if max_loss > 0 else 0

        result = {
            "expiration": best_exp, 
            "dte": dte, 
            "short": short_strike, 
            "long": long_strike,
            "credit": credit, 
            "max_loss": max_loss, 
            "iv": iv,
            "roi": roi
        }
        return result

    except Exception:
        return None

def plot_sparkline_cone(hist, price, iv):
    """Generates the mini chart with expected move cone."""
    fig, ax = plt.subplots(figsize=(4, 1.2)) 
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Last 60 Days History
    last_60 = hist['Close'].tail(60)
    dates = last_60.index
    
    # Projected Cone (30 Days out)
    days_proj = 30
    safe_iv = iv if iv > 0 else 20 
    vol_move = price * (safe_iv/100) * np.sqrt(np.arange(1, days_proj+1)/365)
    
    upper = price + vol_move
    lower = price - vol_move
    future_dates = [dates[-1] + timedelta(days=int(i)) for i in range(1, days_proj+1)]
    
    # Plotting
    ax.plot(dates, last_60, color='#00FFAA', lw=1.2)
    ax.fill_between(future_dates, lower, upper, color='#00FFAA', alpha=0.15)
    ax.plot(future_dates, upper, color='gray', linestyle=':', lw=0.5)
    ax.plot(future_dates, lower, color='gray', linestyle=':', lw=0.5)
    
    ax.axis('off')
    return fig

# --- MAIN UI ---
st.title("🦅 Spread Sniper")
st.markdown("Scanning for **Rank > 80%**, **30 DTE**, **30 Delta** Put Credit Spreads.")

if st.button("🔎 Scan Market (Wide Net)", type="primary"):
    
    status = st.empty()
    progress = st.progress(0)
    
    results = []
    
    # 1. SCAN LOOP
    for i, ticker in enumerate(LIQUID_TICKERS):
        status.caption(f"Analyzing {ticker}...")
        progress.progress((i + 1) / len(LIQUID_TICKERS))
        
        # A. Fetch Data & Calculate Rank
        data = get_stock_data(ticker)
        if not data: continue
        
        # B. STRICT 80% FILTER
        if data['rank'] < 80:
            continue 

        # C. Find Trade Structure
        obj = yf.Ticker(ticker)
        spread = find_credit_spread(obj, data['price'])
        
        if spread:
            # Score = Rank (High is good) + ROI (High is good)
            score = data['rank'] + spread['roi']
            results.append({"ticker": ticker, "data": data, "spread": spread, "score": score})
    
    progress.empty()
    status.empty()
    
    # 2. RESULTS DISPLAY
    if not results:
        st.info("Market is extremely quiet. Even with 60+ tickers, no 80% Rank setups were found.")
        st.caption("Try checking back during market hours or when VIX > 15.")
    else:
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        st.success(f"Found {len(results)} High-Probability Opportunities")
        
        cols = st.columns(3)
        
        for i, res in enumerate(results):
            t = res['ticker']
            d = res['data']
            s = res['spread']
            
            with cols[i % 3]:
                with st.container(border=True):
                    
                    # --- HEADER ---
                    pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="font-size: 22px; font-weight: 900; color: white; line-height: 1;">{t}</div>
                            <div style="margin-top: 4px;"><span class="{pill_class}">${d['price']:.2f} ({d['change_pct']:.2f}%)</span></div>
                        </div>
                        <div class="strategy-badge">PUT SPREAD</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.divider()

                    # --- METRICS GRID ---
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-label">Strikes</div>
                        <div class="metric-value">${s['short']:.0f} / ${s['long']:.0f}</div>
                        <div style="height: 8px;"></div>
                        <div class="metric-label">Credit</div>
                        <div class="metric-value" style="color:#00FFAA">${s['credit']:.2f}</div>
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
                    
                    # --- VISUAL & RANK ---
                    vc1, vc2 = st.columns([2, 1])
                    with vc1:
                         st.pyplot(plot_sparkline_cone(d['hist'], d['price'], s['iv']), use_container_width=True)
                    with vc2:
                        st.markdown(f"""
                        <div class="metric-label" style="text-align: right;">IV Rank</div>
                        <div class="metric-value" style="text-align: right;">{d['rank']:.0f}%</div>
                        <div style="height: 10px;"></div>
                        <div class="metric-label" style="text-align: right;">Opp Score</div>
                        <div class="metric-value" style="text-align: right; color: #d4ac0d;">{res['score']:.0f}</div>
                        """, unsafe_allow_html=True)

                    # --- FOOTER ---
                    st.markdown(f"""
                    <div class="roc-box">
                        <span style="font-size:11px; color: #00c864; text-transform: uppercase; letter-spacing: 1px;">Return on Capital</span><br>
                        <span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Add {t} to Dashboard", key=f"btn_{t}", use_container_width=True):
                        if 'portfolio' not in st.session_state: st.session_state.portfolio = []
                        st.session_state.portfolio.append(res)
                        st.toast(f"{t} added to tracking!")
