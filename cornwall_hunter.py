import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Cornwall Hunter")

# --- CONSTANTS ---
# Use the same list as spread_finder, or a curated list of volatile names
WATCHLIST = [
    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "AMZN", "META", 
    "GOOGL", "NFLX", "COIN", "MSTR", "PLTR", "CRM", "ADBE", "BA", "DIS", "PYPL", 
    "SQ", "SHOP", "ROKU", "DKNG", "UBER", "ABNB", "SNOW", "ZS", "CRWD", "PANW"
]

# --- OPENAI SETUP ---
def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        st.error("Missing OpenAI API Key in secrets.")
        return None

# --- PHASE 1: QUANT SCANNER (The Dragnet) ---
def scan_for_panic(ticker):
    """
    Finds stocks that have dropped >15% in 20 days with high IV.
    Returns dict of data if panic detected, else None.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # 1. The Drop Check (Falling Knife)
        # Find the high in the last 30 days
        last_30 = hist.tail(30)
        high_30d = last_30['High'].max()
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        
        # Filter: Must be down at least 15% from recent high
        if drop_pct > -15.0: return None 
        
        # 2. The Volatility Check
        # Simple Vol Rank approximation
        hist['Returns'] = hist['Close'].pct_change()
        current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
        
        # We want HV to be elevated (indicating panic/activity)
        if current_hv < 30: return None

        return {
            "ticker": ticker,
            "price": current_price,
            "high_30d": high_30d,
            "drop_pct": drop_pct,
            "hv": current_hv,
            "stock_obj": stock
        }
    except: return None

# --- PHASE 2: AI CLASSIFIER (The Judge) ---
def analyze_solvency(client, ticker, stock_obj):
    """
    Fetches news and asks GPT-4o if the drop is terminal or solvable.
    """
    try:
        # 1. Fetch News via yfinance
        news_items = stock_obj.news
        if not news_items: return {"category": "UNKNOWN", "reason": "No news found."}
        
        headlines = [f"- {n['title']}" for n in news_items[:5]] # Top 5 headlines
        news_text = "\n".join(headlines)
        
        # 2. The Prompt
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear.
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary"}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"category": "ERROR", "reason": str(e)}

# --- PHASE 3: THE 10-BAGGER CALCULATOR (The Sniper) ---
def find_cornwall_option(stock_obj, current_price, normal_price):
    """
    Finds a LEAP (1yr+) at the 'Normal Price' strike.
    Checks if potential payout > 10x.
    """
    try:
        # 1. Get Expirations > 300 days
        exps = stock_obj.options
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 300]
        if not leaps: return None
        
        target_exp = leaps[0] # Closest LEAP
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        
        # 2. Find Strike closest to "Normal Price" (Recovery Target)
        # The Cornwall logic: Buy the strike where the stock SHOULD be.
        target_strike = normal_price
        
        # Filter for strikes near target
        candidates = calls[(calls['strike'] >= target_strike * 0.95) & (calls['strike'] <= target_strike * 1.05)]
        if candidates.empty: return None
        
        # Pick the one with liquidity
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        
        ask_price = option['ask']
        if ask_price <= 0: return None
        
        # 3. The Cornwall Ratio
        # If stock goes back to normal, Intrinsic Value = Normal - Strike
        # Payout = Intrinsic / Cost
        projected_intrinsic = normal_price - option['strike']
        if projected_intrinsic <= 0: return None # Target is below strike
        
        payout_ratio = projected_intrinsic / ask_price
        
        return {
            "expiration": target_exp,
            "strike": option['strike'],
            "ask": ask_price,
            "ratio": payout_ratio,
            "contract": option
        }
    except: return None

# --- UI ---
st.title("🏹 Cornwall Hunter")
st.caption("Seeking Solvable Problems Priced as Terminal Risk")

if st.button("Start Hunt"):
    client = get_openai_client()
    if not client: st.stop()
    
    status = st.empty()
    bar = st.progress(0)
    
    found_opportunities = []
    
    for i, ticker in enumerate(WATCHLIST):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(WATCHLIST))
        
        # 1. Quant Filter
        panic_data = scan_for_panic(ticker)
        if not panic_data: continue
        
        status.text(f"💥 Panic detected in {ticker} (-{abs(panic_data['drop_pct']):.1f}%). Analyzing news...")
        
        # 2. AI Judge
        verdict = analyze_solvency(client, ticker, panic_data['stock_obj'])
        
        if verdict.get('category') == 'SOLVABLE':
            # 3. Option Math
            # Assume "Normal" is the 30-day high (Pre-crash level)
            recovery_target = panic_data['high_30d']
            
            opportunity = find_cornwall_option(
                panic_data['stock_obj'], 
                panic_data['price'], 
                recovery_target
            )
            
            if opportunity and opportunity['ratio'] > 5.0: # Filter for at least 5x (relaxed for demo)
                found_opportunities.append({
                    "ticker": ticker,
                    "data": panic_data,
                    "verdict": verdict,
                    "option": opportunity
                })
    
    status.empty()
    bar.empty()
    
    if not found_opportunities:
        st.info("No asymmetric opportunities found in the current watchlist.")
    else:
        st.success(f"Found {len(found_opportunities)} Cornwall Setups")
        
        for opp in found_opportunities:
            t = opp['ticker']
            d = opp['data']
            v = opp['verdict']
            o = opp['option']
            
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 4, 2])
                
                with c1:
                    st.metric(label=t, value=f"${d['price']:.2f}", delta=f"{d['drop_pct']:.1f}% (30d)")
                    st.caption(f"Target: ${d['high_30d']:.2f}")
                
                with c2:
                    st.subheader("💡 Analysis: SOLVABLE")
                    st.write(f"_{v['reason']}_")
                    st.markdown(f"**Catalyst:** Market overreaction to fixable news.")
                
                with c3:
                    st.metric(label="Potential Payout", value=f"{o['ratio']:.1f}x")
                    st.markdown(f"**Buy:** {o['expiration']} **${o['strike']:.0f} Call**")
                    st.markdown(f"**Cost:** ${o['ask']:.2f}")
