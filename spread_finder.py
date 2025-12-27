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
        st.warning("⚠️ Google API Key not found in secrets.")
except Exception as e:
    st.warning(f"⚠️ AI Setup Error: {e}")

# 2. Universe (Liquid S&P 500 Proxies)
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", 
    "GOOGL", "META", "NFLX", "BAC", "JPM", "XOM", "CVX", "DIS", "BA", "PLTR", "COIN"
]

# --- BLACK-SCHOLES IV CALCULATOR ---

def newton_vol_call(S, K, T, C, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    return sigma - fx / vega

def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    return sigma - fx / vega

def get_implied_volatility(option_type, price, strike, time_to_exp, market_price, risk_free_rate=0.045):
    sigma = 0.5 
    try:
        for i in range(100):
            if option_type == 'call':
                diff = newton_vol_call(price, strike, time_to_exp, market_price, risk_free_rate, sigma)
            else:
                diff = newton_vol_put(price, strike, time_to_exp, market_price, risk_free_rate, sigma)
            sigma = diff
            if abs(diff - sigma) < 1e-5:
                break
        return abs(sigma)
    except:
        return 0.0

# --- DATA & ANALYSIS ---

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    """
    Fetches price history and calculates Realized Volatility Rank.
    IMPORTANT: Do not return the 'yf.Ticker' object here, it breaks caching.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None

        current_price = hist['Close'].iloc[-1]
        
        # 1. Calculate Realized Volatility (HV)
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        
        current_vol = hist['HV'].iloc[-1]
        min_vol = hist['HV'].min()
        max_vol = hist['HV'].max()
        
        if max_vol == min_vol:
            rank = 50
        else:
            rank = ((current_vol - min_vol) / (max_vol - min_vol)) * 100
            
        # Get Next Earnings (With robust error handling)
        earnings_date = None
        try:
            cal = stock.calendar
            if cal is not None and not cal.empty:
                # yfinance calendar format varies, usually row 0 is next earnings
                potential_date = cal.iloc[0][0]
                # Check if it's a list or single value
                if isinstance(potential_date, (list, tuple)):
                    future = [d for d in potential_date if d > datetime.now().date()]
                    if future: earnings_date = future[0]
                elif hasattr(potential_date, 'date'):
                     if potential_date.date() > datetime.now().date():
                        earnings_date = potential_date.date()
        except Exception:
            pass # Earnings data failure shouldn't crash the scanner

        return {
            "price": current_price,
            "rank": rank,
            "earnings": earnings_date,
            "hist": hist,
            # "obj": stock  <-- REMOVED to fix pickle error
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def find_credit_spread(stock_obj, current_price, target_dte=30, width=5.0):
    """Scans for the 30-delta spread and calculates accurate IV using BS."""
    
    # 1. Get Expiration
    try:
        exps = stock_obj.options
        if not exps: return None
    except Exception:
        return None
        
    target_date = datetime.now() + timedelta(days=target_dte)
    best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
    days_to_exp = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
    
    if days_to_exp < 7: return None 

    # 2. Get Chain
    try:
        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
    except:
        return None

    # 3. Find Short Leg (~30 Delta Proxy: Strike ~96% of Price)
    target_strike = current_price * 0.96
    
    short_leg = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]
    if short_leg.empty: return None
    
    short_strike = short_leg.iloc[0]['strike']
    short_bid = short_leg.iloc[0]['bid']
    short_ask = short_leg.iloc[0]['ask']
    short_mid = (short_bid + short_ask) / 2
    
    # 4. Find Long Leg
    long_strike_target = short_strike - width
    long_leg = puts.iloc[(puts['strike'] - long_strike_target).abs().argsort()[:1]]
    if long_leg.empty: return None
        
    long_strike = long_leg.iloc[0]['strike']
    long_ask = long_leg.iloc[0]['ask']
    
    actual_width = short_strike - long_strike
    if abs(actual_width - width) > 1.0: return None

    # 5. Calculate Exact IV
    time_years = days_to_exp / 365.0
    bs_iv = get_implied_volatility('put', current_price, short_strike, time_years, short_mid)

    credit = short_bid - long_ask
    
    return {
        "expiration": best_exp,
        "dte": days_to_exp,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "credit": credit,
        "iv": bs_iv * 100, 
        "roi": (credit / (actual_width - credit)) * 100 if (actual_width - credit) > 0 else 0
    }

def get_ai_analysis(ticker, rank, earnings_days):
    prompt = f"""
    Analyze Put Credit Spread on {ticker}.
    Context:
    - Volatility Rank is {rank:.0f}% (Elevated).
    - Earnings in {earnings_days} days.
    
    Return JSON:
    {{
        "sentiment": "Bullish/Bearish/Neutral",
        "verdict": "Safe/Speculative/Avoid",
        "explanation": "One concise sentence on why IV is high."
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "")
        return eval(clean) 
    except:
        return {"verdict": "Unknown", "explanation": "AI Service unavailable."}

# --- MATPLOTLIB VISUALIZATION ---

def plot_cone_matplotlib(hist_data, current_price, ticker, iv):
    prices = hist_data['Close'].tail(90)
    dates = prices.index
    
    days_proj = 30
    proj_days = np.arange(1, days_proj + 1)
    
    # Check if IV is valid (sometimes calculation fails and returns 0)
    safe_iv = iv if iv > 1.0 else 20.0 # Default to 20% if 0
    
    vol_move = current_price * (safe_iv / 100) * np.sqrt(proj_days / 365)
    
    upper_cone = current_price + vol_move
    lower_cone = current_price - vol_move
    
    last_date = dates[-1]
    future_dates = [last_date + timedelta(days=int(d)) for d in proj_days]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.plot(dates, prices, color='#00FFAA', linewidth=1.5, label='History')
    ax.plot(future_dates, upper_cone, color='white', linestyle=':', alpha=0.5)
    ax.plot(future_dates, lower_cone, color='white', linestyle=':', alpha=0.5)
    ax.fill_between(future_dates, lower_cone, upper_cone, color='#00FFAA', alpha=0.15)
    
    ax.set_title(f"{ticker} | Implied Vol: {safe_iv:.1f}%", color='white', fontsize=10)
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#333')
    ax.spines['top'].set_color('#333') 
    ax.spines['left'].set_color('#333')
    ax.spines['right'].set_color('#333')
    ax.grid(color='#333', linestyle='--', linewidth=0.5)
    
    return fig

# --- MAIN PAGE ---

st.title("🦅 Spread Finder: Black-Scholes Edition")

with st.sidebar:
    st.header("Settings")
    min_rank = st.slider("Min Vol Rank", 0, 100, 50)
    simulate_data = st.checkbox("Simulate High Rank", value=True, help="Injects fake high rank for demo.")

if st.button("🔎 Scan Liquid Tickers", type="primary"):
    
    prog = st.progress(0)
    status = st.empty()
    opportunities = []
    
    for i, ticker in enumerate(LIQUID_TICKERS):
        status.text(f"Processing {ticker}...")
        prog.progress((i + 1) / len(LIQUID_TICKERS))
        
        # 1. Get Cached Data (Pure data only)
        data = get_stock_data(ticker)
        if not data: continue
        
        # 2. Re-create Ticker Object for Live Calls (cannot be cached)
        # This is fast, as it doesn't fetch data until we ask for options
        stock_obj = yf.Ticker(ticker)

        # 3. Filter by Rank
        rank = data['rank']
        if simulate_data: rank = np.random.randint(60, 95)
        
        if rank < min_rank: continue
        
        # 4. Find Spread & Calc IV (Pass the fresh object)
        spread = find_credit_spread(stock_obj, data['price'])
        
        if spread:
            earnings = 999
            if data['earnings']:
                if isinstance(data['earnings'], (datetime, pd.Timestamp)):
                     earnings = (data['earnings'].date() - datetime.now().date()).days
                else:
                     earnings = (pd.to_datetime(data['earnings']).date() - datetime.now().date()).days
            
            # 5. AI Check
            ai = get_ai_analysis(ticker, rank, earnings)
            
            # 6. Score
            score = 0
            if rank > 70: score += 30
            if spread['roi'] > 10: score += 20
            if ai.get('verdict') == "Safe": score += 30
            if earnings > 35: score += 20
            
            opportunities.append({
                "ticker": ticker,
                "score": score,
                "rank": rank,
                "price": data['price'],
                "spread": spread,
                "ai": ai,
                "hist": data['hist']
            })
            
    prog.empty()
    status.empty()
    
    if opportunities:
        opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)
        st.success(f"Found {len(opportunities)} Trades")
        
        for opp in opportunities:
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 2])
                with c1: 
                    st.subheader(opp['ticker'])
                    st.caption(f"${opp['price']:.2f}")
                with c2:
                    st.write(f"**Score: {opp['score']}**")
                    st.progress(opp['score']/100)
                with c3:
                    if st.button(f"Add {opp['ticker']}", key=opp['ticker']):
                        if 'portfolio' not in st.session_state: st.session_state.portfolio = []
                        st.session_state.portfolio.append(opp)
                        st.toast("Added!")
                
                st.divider()
                
                # Details
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.markdown("#### Volatility")
                    st.metric("Vol Rank (Year)", f"{opp['rank']:.0f}%")
                    st.metric("Implied Vol (Live)", f"{opp['spread']['iv']:.1f}%")
                with dc2:
                    st.markdown("#### Trade")
                    st.text(f"Short: ${opp['spread']['short_strike']}")
                    st.text(f"Long:  ${opp['spread']['long_strike']}")
                    st.text(f"Credit: ${opp['spread']['credit']:.2f}")
                with dc3:
                    st.markdown("#### AI Verdict")
                    st.info(f"{opp['ai']['verdict']}: {opp['ai']['explanation']}")
                
                # Matplotlib Chart
                st.pyplot(plot_cone_matplotlib(opp['hist'], opp['price'], opp['ticker'], opp['spread']['iv']))

    else:
        st.info("No trades found.")
