import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

# --- SILENCE WARNINGS ---
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- GOOGLE AI LIBRARY ---
import google.generativeai as genai

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
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (Gemini 2.0)</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Test Logic)", value=False, help="Forces detection of 'Panic' to test AI and UI logic.")
    
    # DEBUG TOOL
    if st.button("🔍 Check Available Models"):
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success("✅ Valid Models Found:")
                st.code("\n".join(models))
            else:
                st.error("No API Key found.")
        except Exception as e:
            st.error(f"Error checking models: {e}")

# --- EXPANDED UNIVERSE (Mid-Caps, High-Beta, Crypto, Bio) ---
LIQUID_TICKERS = [
    # --- HIGH BETA / GROWTH ---
    "PLTR", "SOFI", "HOOD", "DKNG", "ROKU", "SHOP", "SQ", "AFRM", "UPST", "CVNA",
    "NET", "DDOG", "SNOW", "U", "RBLX", "COIN", "MSTR", "MARA", "RIOT", "CLSK",
    # --- VOLATILE TECH & AI ---
    "AI", "IONQ", "PLUG", "FCEL", "JOBY", "ACHR", "SPCE", "ASTS", "LUNR",
    # --- BIOTECH (Binary Events) ---
    "XBI", "LABU", "MRNA", "BNTX", "CRSP", "NTLA", "EDIT", "BEAM",
    # --- MEME / RETAIL ---
    "GME", "AMC", "CHWY", "RDDT", "DJT",
    # --- DISASTER RECOVERY (Legacy) ---
    "BA", "INTC", "WBA", "PARA", "LULU", "NKE", "SBUX", "PYPL",
    # --- INDEXES ---
    "SPY", "QQQ", "IWM", "ARKK", "KRE"
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
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # --- CRITICAL FIX: FORCE PASS IN DEV MODE ---
        if dev:
            # We ignore real data and fabricate a "Perfect Setup"
            # This guarantees it passes the -15% filter below
            return {
                "ticker": ticker,
                "price": current_price,
                "high_30d": current_price * 1.40, # Fake High (Implies 30% drop)
                "drop_pct": -30.0, # Fake Drop
                "hv": 85.0,        # Fake Volatility
                "stock_obj": stock
            }

        # --- NORMAL PRODUCTION LOGIC ---
        last_30 = hist.tail(30)
        high_30d = last_30['High'].max()
        if high_30d <= 0: return None
        
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        hist['Returns'] = hist['Close'].pct_change()
        current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
        
        # Production Filters: >15% drop, >35 HV
        if drop_pct > -15.0: return None 
        if current_hv < 35: return None 

        return {
            "ticker": ticker,
            "price": current_price,
            "high_30d": high_30d,
            "drop_pct": drop_pct,
            "hv": current_hv,
            "stock_obj": stock
        }
    except: return None

# --- PHASE 2: AI CLASSIFIER ---
def analyze_solvency_gemini(ticker, stock_obj, dev=False):
    try:
        news_items = stock_obj.news
        if not news_items:
            if dev: return {"category": "SOLVABLE", "reason": "(DEV) No news, forced solvable."}
            return {"category": "UNKNOWN", "reason": "No news found."}
        
        headlines = []
        for n in news_items[:5]:
            title = n.get('title', n.get('headline', 'No Title'))
            headlines.append(f"- {title}")
        news_text = "\n".join(headlines)
        
        model = genai.GenerativeModel('gemini-2.0-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete, Dilution Spiral.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear, Short Report.
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary (max 15 words)"}}
        """
        
        response = model.generate_content(prompt)
        return json.loads(response.text)
        
    except Exception as e:
        if "429" in str(e): return {"category": "RATE_LIMIT", "reason": "Gemini Rate Limit Hit."}
        return {"category": "ERROR", "reason": str(e)}

# --- PHASE 3: OPTION MATH ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    try:
        exps = stock_obj.options
        if not exps: return None
        
        # LEAPS > 250 days
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 250]
        if not leaps: return None
        
        target_exp = leaps[0] 
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        
        target_strike = normal_price
        
        # Search Range: Production (5%), Dev (Huge 50% range to force a hit)
        candidates = calls[(calls['strike'] >= target_strike * 0.95) & (calls['strike'] <= target_strike * 1.05)]
        
        if candidates.empty and dev:
             candidates = calls[(calls['strike'] >= target_strike * 0.50) & (calls['strike'] <= target_strike * 1.50)]

        if candidates.empty: return None
        
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        ask_price = option['ask']
        
        # Fix Bad Data in Dev
        if ask_price <= 0:
            if dev: ask_price = 0.50
            else: return None
        
        projected_intrinsic = normal_price - option['strike']
        
        # Fix Negative Math in Dev
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
    
    scan_list = LIQUID_TICKERS[:3] if dev_mode else LIQUID_TICKERS
    
    if "last_gemini_call" not in st.session_state:
        st.session_state.last_gemini_call = 0
    
    RATE_LIMIT_DELAY = 5.0 

    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # --- QUANT (Force Pass in Dev) ---
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Analyzing news...")
        
        # --- AI LOOP ---
        verdict = None
        retries = 0
        while retries < 3:
            # 1. Gatekeeper
            elapsed = time.time() - st.session_state.last_gemini_call
            if elapsed < RATE_LIMIT_DELAY:
                sleep_time = RATE_LIMIT_DELAY - elapsed
                if sleep_time > 1.0:
                    status.text(f"⏳ Gatekeeper: Pausing {sleep_time:.1f}s for API limit...")
                time.sleep(sleep_time)

            # 2. Call
            result = analyze_solvency_gemini(ticker, panic_data['stock_obj'], dev=dev_mode)
            st.session_state.last_gemini_call = time.time()
            
            # 3. Check
            if result.get('category') == "RATE_LIMIT":
                status.warning(f"⚠️ Rate Limit (429). Auto-cooling 60s... ({retries+1}/3)")
                time.sleep(60)
                retries += 1
            else:
                verdict = result
                break
        
        if not verdict or verdict.get('category') == "RATE_LIMIT":
            status.error(f"Skipping {ticker} due to API limits.")
            continue

        # --- RESULTS ---
        if verdict.get('category') == 'SOLVABLE' or (dev_mode and verdict.get('category') != 'ERROR'):
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
                
                if dev_mode:
                    status.text("✅ Dev Mode: Opportunity found. Stopping scan.")
                    break
    
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
