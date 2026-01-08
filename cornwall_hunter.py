import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Cornwall Hunter")

# --- UI HEADER (MATCHING DASHBOARD) ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Cornwall Hunter</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR & DEV MODE ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Test Logic)", value=False, help="Lowers drop threshold from 15% to 1% to find test candidates.")

# --- CONSTANTS (MATCHING SPREAD FINDER) ---
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

# --- OPENAI SETUP ---
def get_openai_client():
    # Try fetching from top level secrets first, then google_oauth section fallback
    key = st.secrets.get("OPENAI_API_KEY")
    
    if not key:
        st.error("Missing OPENAI_API_KEY in secrets.toml")
        return None
        
    return OpenAI(api_key=key)

# --- PHASE 1: QUANT SCANNER (The Dragnet) ---
def scan_for_panic(ticker, dev=False):
    """
    Finds stocks that have dropped significantly.
    Normal Mode: >15% drop in 20 days.
    Dev Mode: >1% drop in 20 days (to find matches for testing).
    """
    try:
        stock = yf.Ticker(ticker)
        # Fetch 3mo to ensure we have enough data for 20d calc + buffer
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # 1. The Drop Check (Falling Knife)
        # Find the high in the last 30 days
        last_30 = hist.tail(30)
        high_30d = last_30['High'].max()
        
        if high_30d <= 0: return None
        
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        
        # THRESHOLD LOGIC
        threshold = -1.0 if dev else -15.0
        
        if drop_pct > threshold: return None 
        
        # 2. The Volatility Check
        # Simple Vol Rank approximation
        hist['Returns'] = hist['Close'].pct_change()
        current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
        
        # We want HV to be elevated (indicating panic/activity)
        hv_threshold = 10 if dev else 30
        if current_hv < hv_threshold: return None

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
def analyze_solvency(client, ticker, stock_obj, dev=False):
    """
    Fetches news and asks GPT-4o if the drop is terminal or solvable.
    """
    try:
        # 1. Fetch News via yfinance
        news_items = stock_obj.news
        
        # Fallback for no news in Dev Mode
        if not news_items:
            if dev:
                return {
                    "category": "SOLVABLE", 
                    "reason": "(DEV MODE) No news found, forcing Solvable verdict for testing."
                }
            return {"category": "UNKNOWN", "reason": "No news found."}
        
        headlines = [f"- {n['title']}" for n in news_items[:5]] 
        news_text = "\n".join(headlines)
        
        # 2. The Prompt
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear, General Market Correction.
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary (max 15 words)"}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"category": "ERROR", "reason": str(e)}

# --- PHASE 3: THE 10-BAGGER CALCULATOR (The Sniper) ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
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
        
        # If strict target missing, widen search in dev mode
        if candidates.empty and dev:
             candidates = calls[(calls['strike'] >= target_strike * 0.80) & (calls['strike'] <= target_strike * 1.20)]

        if candidates.empty: return None
        
        # Pick the one with liquidity
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        
        ask_price = option['ask']
        if ask_price <= 0: return None
        
        # 3. The Cornwall Ratio
        # If stock goes back to normal, Intrinsic Value = Normal - Strike
        # Payout = Intrinsic / Cost
        projected_intrinsic = normal_price - option['strike']
        
        # If Target is below Strike, intrinsic is 0. 
        # In Dev Mode, we allow this just to show the card.
        if projected_intrinsic <= 0:
            if dev: projected_intrinsic = 0.5 # Fake intrinsic for math
            else: return None 
        
        payout_ratio = projected_intrinsic / ask_price
        
        return {
            "expiration": target_exp,
            "strike": option['strike'],
            "ask": ask_price,
            "ratio": payout_ratio,
            "contract": option
        }
    except: return None

# --- MAIN EXECUTION ---

if st.button(f"Start Hunt {'(Dev Mode)' if dev_mode else ''}"):
    client = get_openai_client()
    if not client: st.stop()
    
    status = st.empty()
    bar = st.progress(0)
    
    found_opportunities = []
    
    # In Dev Mode, limit to first 10 tickers to save time/API calls
    scan_list = LIQUID_TICKERS[:15] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # 1. Quant Filter
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic detected in {ticker} (-{abs(panic_data['drop_pct']):.1f}%). Analyzing news...")
        
        # 2. AI Judge
        verdict = analyze_solvency(client, ticker, panic_data['stock_obj'], dev=dev_mode)
        
        # In Dev Mode, we accept everything just to show the UI
        if verdict.get('category') == 'SOLVABLE' or dev_mode:
            # 3. Option Math
            # Assume "Normal" is the 30-day high (Pre-crash level)
            recovery_target = panic_data['high_30d']
            
            opportunity = find_cornwall_option(
                panic_data['stock_obj'], 
                panic_data['price'], 
                recovery_target,
                dev=dev_mode
            )
            
            # Filter Logic
            min_ratio = 2.0 if dev_mode else 8.0
            
            if opportunity and opportunity['ratio'] > min_ratio:
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
                # Header
                st.markdown(f"### {t} <span style='font-size:16px; color:#ff4b4b;'>{d['drop_pct']:.1f}% Drop</span>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([2, 4, 2])
                
                with c1:
                    st.metric(label="Current Price", value=f"${d['price']:.2f}")
                    st.caption(f"Recovery Target: ${d['high_30d']:.2f}")
                
                with c2:
                    category_color = "#00C853" if v.get('category') == 'SOLVABLE' else "#FFA726"
                    st.markdown(f"**AI Verdict:** <span style='color:{category_color}; font-weight:bold;'>{v.get('category', 'UNKNOWN')}</span>", unsafe_allow_html=True)
                    st.write(f"_{v.get('reason', 'No reason provided')}_")
                
                with c3:
                    st.metric(label="Potential Payout", value=f"{o['ratio']:.1f}x")
                    st.markdown(f"**Buy:** {o['expiration']} **${o['strike']:.0f} Call**")
                    st.markdown(f"**Cost:** ${o['ask']:.2f}")
