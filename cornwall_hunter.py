import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
import json

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Cornwall Hunter")

# --- UI HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Cornwall Hunter</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (Gemini Powered)</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Test Logic)", value=False, help="Forces detection of 'Panic' to test AI and UI logic.")

# --- CONSTANTS ---
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

# --- GEMINI SETUP ---
def configure_gemini():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in secrets.toml")
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Gemini Config Error: {e}")
        return False

# --- PHASE 1: QUANT SCANNER ---
def scan_for_panic(ticker, dev=False):
    """
    Finds stocks that have dropped significantly.
    Normal Mode: >15% drop in 20 days.
    Dev Mode: FORCE PASS by simulating panic data if real data is healthy.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # Drop Check
        last_30 = hist.tail(30)
        high_30d = last_30['High'].max()
        if high_30d <= 0: return None
        
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        
        # Volatility Check
        hist['Returns'] = hist['Close'].pct_change()
        current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
        
        # --- DEV MODE OVERRIDE ---
        if dev:
            # If stock is "Healthy" (e.g. down less than 5%), FAKE IT.
            if drop_pct > -5.0:
                return {
                    "ticker": ticker,
                    "price": current_price,
                    "high_30d": current_price * 1.25, # Fake High (implying 20% drop)
                    "drop_pct": -20.0, # Fake 20% Drop
                    "hv": 85.0,        # Fake High Vol
                    "stock_obj": stock
                }
        
        # --- PRODUCTION LOGIC ---
        if drop_pct > -15.0: return None 
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

# --- PHASE 2: AI CLASSIFIER (GEMINI) ---
def analyze_solvency_gemini(ticker, stock_obj, dev=False):
    """
    Uses Google Gemini Flash to classify the drop.
    """
    try:
        # 1. Fetch News
        news_items = stock_obj.news
        
        if not news_items:
            if dev: return {"category": "SOLVABLE", "reason": "(DEV) No news, forced solvable."}
            return {"category": "UNKNOWN", "reason": "No news found."}
        
        headlines = []
        for n in news_items[:5]:
            title = n.get('title', n.get('headline', 'No Title'))
            headlines.append(f"- {title}")
        news_text = "\n".join(headlines)
        
        # 2. Configure Model
        # Using gemini-1.5-flash for speed and low cost (free tier)
        model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 3. Prompt
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear.
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary (max 15 words)"}}
        """
        
        response = model.generate_content(prompt)
        return json.loads(response.text)
        
    except Exception as e:
        # Check for specific Gemini errors (like blocked content)
        if "429" in str(e):
            return {"category": "RATE_LIMIT", "reason": "Gemini Rate Limit Hit."}
        return {"category": "ERROR", "reason": str(e)}

# --- PHASE 3: OPTION MATH ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    try:
        exps = stock_obj.options
        if not exps: return None
        
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 300]
        if not leaps: return None
        
        target_exp = leaps[0] 
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        
        target_strike = normal_price
        candidates = calls[(calls['strike'] >= target_strike * 0.95) & (calls['strike'] <= target_strike * 1.05)]
        
        if candidates.empty and dev:
             candidates = calls[(calls['strike'] >= target_strike * 0.80) & (calls['strike'] <= target_strike * 1.20)]

        if candidates.empty: return None
        
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        ask_price = option['ask']
        
        if ask_price <= 0:
            if dev: ask_price = 0.50
            else: return None
        
        projected_intrinsic = normal_price - option['strike']
        
        if projected_intrinsic <= 0:
            if dev: projected_intrinsic = 5.0 
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
    if not configure_gemini(): st.stop()
    
    status = st.empty()
    bar = st.progress(0)
    found_opportunities = []
    
    scan_list = LIQUID_TICKERS[:10] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # 1. Quant
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Asking Gemini...")
        
        # 2. AI (Gemini)
        verdict = analyze_solvency_gemini(ticker, panic_data['stock_obj'], dev=dev_mode)
        
        if verdict.get('category') == "RATE_LIMIT":
            st.warning("Gemini Rate Limit reached. Pausing...")
            break
            
        if verdict.get('category') == 'SOLVABLE' or dev_mode:
            # 3. Math
            recovery_target = panic_data['high_30d']
            opportunity = find_cornwall_option(
                panic_data['stock_obj'], 
                panic_data['price'], 
                recovery_target,
                dev=dev_mode
            )
            
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
        st.info("No asymmetric opportunities found.")
    else:
        st.success(f"Found {len(found_opportunities)} Cornwall Setups")
        
        for opp in found_opportunities:
            t = opp['ticker']
            d = opp['data']
            v = opp['verdict']
            o = opp['option']
            
            with st.container(border=True):
                st.markdown(f"### {t} <span style='font-size:16px; color:#ff4b4b;'>{d['drop_pct']:.1f}% Drop</span>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([2, 4, 2])
                with c1:
                    st.metric("Price", f"${d['price']:.2f}")
                    st.caption(f"Target: ${d['high_30d']:.2f}")
                with c2:
                    cat = v.get('category', 'UNKNOWN')
                    color = "#00C853" if cat == 'SOLVABLE' else "#FFA726"
                    st.markdown(f"**Gemini:** <span style='color:{color}; font-weight:bold;'>{cat}</span>", unsafe_allow_html=True)
                    st.write(f"_{v.get('reason')}_")
                with c3:
                    st.metric("Payout", f"{o['ratio']:.1f}x")
                    st.markdown(f"**{o['expiration']} ${o['strike']:.0f} C**")
                    st.markdown(f"**Cost:** ${o['ask']:.2f}")
