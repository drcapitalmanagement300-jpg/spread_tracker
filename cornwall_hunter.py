import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import requests
import xml.etree.ElementTree as ET
import os
import random

# --- SILENCE WARNINGS ---
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Cornwall Hunter")

# --- UI HEADER ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    logo_path = "754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG"
    if os.path.exists(logo_path):
        st.image(logo_path, width=130)
    else:
        st.write("**DR CAPITAL**")
        
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Cornwall Hunter</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (Diagnostic V34)</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Diagnostic)", value=False, help="Forces a trade to appear. Prints debug steps to screen.")
    
    st.divider()
    st.write("### 🔌 Connection Debugger")
    
    if st.button("Test Hugging Face Router"):
        api_key = st.secrets.get("HUGGINGFACE_API_KEY")
        if not api_key:
            st.error("Missing HUGGINGFACE_API_KEY")
        else:
            try:
                # Use Llama 3.2 (Supported on Chat Router)
                API_URL = "https://router.huggingface.co/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": "meta-llama/Llama-3.2-3B-Instruct",
                    "messages": [{"role": "user", "content": "Reply with one word: Connected"}],
                    "max_tokens": 10
                }
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    msg = data['choices'][0]['message']['content']
                    st.success(f"✅ Success: {msg}")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

# --- EXPANDED UNIVERSE ---
LIQUID_TICKERS = list(set([
    "TSLA", "NVDA", "AMD", "AMZN", "GOOGL", "META", "MSFT", "NFLX",
    "PLTR", "SOFI", "HOOD", "DKNG", "ROKU", "SHOP", "SQ", "AFRM", "UPST", "CVNA",
    "NET", "DDOG", "SNOW", "U", "RBLX", "COIN", "CRWD", "ZS", "PANW", "TTD",
    "APP", "MDB", "TEAM", "HUBS", "BILL", "DOCU", "TWLO", "OKTA", "ZM",
    "MSTR", "MARA", "RIOT", "CLSK", "CIFR", "IREN", "WULF", "HUT", "BITF",
    "SMCI", "ARM", "MU", "INTC", "QCOM", "AVGO", "MRVL", "ANET", "VRT",
    "BA", "WBA", "DIS", "NKE", "SBUX", "LULU", "EL", "ULTA", "DG", "DLTR",
    "MMM", "T", "VZ", "PFE", "BMY", "CVS",
    "XBI", "LABU", "MRNA", "BNTX", "CRSP", "NTLA", "EDIT", "BEAM", "SAVA",
    "CCL", "RCL", "NCLH", "UAL", "AAL", "DAL", "LUV", "EXPE", "ABNB", "BKNG",
    "LVS", "WYNN", "MGM", "CZR", "PENN",
    "URA", "CCJ", "FCX", "SCCO", "CLF", "AA",
    "CHWY", "RDDT", "DJT", "SPCE", "ASTS", "LUNR", "RKLB", "JOBY", "ACHR",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "ARKK", "KRE", "XOP", "TAN"
]))

# --- DEBUG LOGGER ---
def log_debug(msg):
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

# --- HELPER: GOOGLE NEWS RSS FETCHER ---
def get_google_news(ticker):
    try:
        query = f"{ticker} stock news"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        if response.status_code != 200: return ""
        root = ET.fromstring(response.content)
        headlines = []
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text
            headlines.append(f"- {title}")
        return "\n".join(headlines)
    except Exception as e:
        log_debug(f"Google News Fetch Failed for {ticker}: {e}")
        return ""

# --- SESSION ROTATOR (ANTI-BAN) ---
def get_fresh_session():
    """Generates a new session with random User-Agent to trick Yahoo."""
    session = requests.Session()
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ]
    session.headers.update({"User-Agent": random.choice(user_agents)})
    return session

# --- PHASE 1: QUANT SCANNER ---
def scan_for_panic(ticker, session, dev=False):
    # --- CRITICAL FIX: ISOLATE DEV MODE ---
    # Return immediately. Do not touch yfinance.
    if dev:
        # Create a dummy stock object wrapper to prevent errors downstream
        class MockStock:
            def __init__(self): self.news = []
        
        return {
            "ticker": ticker,
            "price": 100.0,
            "high_30d": 145.0, 
            "drop_pct": -30.0,
            "hv": 85.0,
            "stock_obj": MockStock()
        }

    max_retries = 2
    wait_time = 5
    
    for attempt in range(max_retries + 1):
        try:
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                if attempt < max_retries:
                    log_debug(f"⚠️ {ticker} empty data. Retrying {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    wait_time *= 2
                    continue
                else:
                    return None
            
            current_price = hist['Close'].iloc[-1]
            last_30 = hist.tail(30)
            high_30d = last_30['High'].max()
            if high_30d <= 0: return None
            
            drop_pct = ((current_price - high_30d) / high_30d) * 100
            hist['Returns'] = hist['Close'].pct_change()
            current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
            
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

        except Exception as e:
            if attempt < max_retries:
                time.sleep(wait_time)
                continue
            return None

# --- PHASE 2: AI CLASSIFIER (UNIVERSAL ROUTER) ---
def analyze_solvency_hf(ticker, stock_obj, api_key, dev=False):
    try:
        # --- DEV MODE: REAL AI, MOCK NEWS ---
        if dev:
            news_text = "- CEO unexpectedly resigns citing personal reasons.\n- Q3 earnings missed estimates by 5%.\n- Analysts maintain Buy rating despite volatility."
            source_used = "Dev Mode Mock News (Testing AI Connection...)"
        else:
            # 1. Google News Primary
            news_text = get_google_news(ticker)
            source_used = "Google News RSS"
            
        if not news_text:
            return {"category": "UNKNOWN", "reason": "No news found."}
        
        # 2. Universal Router Call
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are a financial risk analyst.
        Analyze these headlines for {ticker}:
        {news_text}
        
        Is this a TERMINAL problem (Fraud, Bankruptcy) or SOLVABLE (Earnings Miss, Macro)?
        
        Respond with valid JSON only: {{"category": "TERMINAL" or "SOLVABLE", "reason": "summary"}}"""
        
        MODELS_TO_TRY = [
            "meta-llama/Llama-3.2-3B-Instruct", 
            "microsoft/Phi-3.5-mini-instruct",
            "HuggingFaceH4/zephyr-7b-beta"
        ]
        
        for model_id in MODELS_TO_TRY:
            try:
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.1
                }
                
                for _ in range(2):
                    response = requests.post(API_URL, headers=headers, json=payload)
                    if response.status_code == 200:
                        break
                    time.sleep(2)
                
                if response.status_code != 200:
                    continue 
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                content = content.replace("```json", "").replace("```", "").strip()
                
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    parsed['sources'] = f"**Source: {source_used}**\n\n{news_text}"
                    return parsed
            
            except:
                continue
        
        # If we reach here, models failed.
        # DEV MODE OVERRIDE: If AI fails but we are in dev mode, return a dummy success
        if dev:
            return {"category": "SOLVABLE", "reason": "Dev Mode Force Pass (Connection Glitch)", "sources": "Simulated AI Response"}
            
        return {"category": "ERROR", "reason": "AI Models Unavailable"}
            
    except:
        if dev:
            return {"category": "SOLVABLE", "reason": "Dev Mode Force Pass (Critical Exception)", "sources": "Simulated AI Response"}
        return {"category": "ERROR", "reason": "AI Critical Failure"}

# --- PHASE 3: OPTION MATH ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    # --- CRITICAL FIX: ISOLATE DEV MODE ---
    if dev:
        return {
            "expiration": "2027-01-15",
            "strike": round(normal_price, 0),
            "ask": 0.50,
            "ratio": 20.0,
            "contract": {"strike": normal_price, "ask": 0.50}
        }

    try:
        exps = stock_obj.options
        if not exps: return None
        
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 360]
        if not leaps: return None
        
        target_exp = leaps[-1] 
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        target_strike = normal_price
        
        candidates = calls[(calls['strike'] >= target_strike * 0.80) & (calls['strike'] <= target_strike * 1.20)].copy()
        
        def get_valid_price(row):
            ask = row.get('ask', 0)
            bid = row.get('bid', 0)
            last = row.get('lastPrice', 0)
            if ask > 0: return ask
            if bid > 0 and last > 0: return (bid + last) / 2
            return last

        candidates['valid_price'] = candidates.apply(get_valid_price, axis=1)
        candidates = candidates[candidates['valid_price'] > 0]
        
        if candidates.empty: return None
            
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        ask_price = option['valid_price']
        projected_intrinsic = normal_price - option['strike']
        
        if projected_intrinsic <= 0: return None
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
    api_key = st.secrets.get("HUGGINGFACE_API_KEY")
    if not api_key:
        st.error("Missing HUGGINGFACE_API_KEY in secrets.toml")
        st.stop()
    
    status = st.empty()
    bar = st.progress(0)
    st.session_state.debug_logs = [] 
    found_opportunities = []
    
    current_session = get_fresh_session()
    scan_list = LIQUID_TICKERS[:1] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        if not dev_mode:
            if i > 0 and i % 10 == 0:
                current_session = get_fresh_session()
                time.sleep(1.0)
            time.sleep(1.0) 
        
        # 1. QUANT
        panic_data = scan_for_panic(ticker, session=current_session, dev=dev_mode)
        if not panic_data: 
            if dev_mode: st.warning(f"Dev Mode: Quant Step Failed for {ticker}")
            continue
        
        if dev_mode: st.text(f"🔹 Quant Passed: {ticker}")
        else: status.text(f"💥 Panic: {ticker}. Intel gathering...")
        
        # 2. AI
        if not dev_mode:
            time.sleep(0.5) 
        
        verdict = analyze_solvency_hf(ticker, panic_data['stock_obj'], api_key, dev=dev_mode)
        
        if not verdict or verdict.get('category') == "ERROR":
            if dev_mode: 
                # This should be unreachable due to overrides, but just in case:
                st.warning(f"Dev Mode: AI Step Failed for {ticker}")
            else:
                continue
        
        if dev_mode: st.text(f"🔹 AI Passed: {verdict.get('category')}")

        # 3. MATH
        # FORCE PASS IN DEV MODE
        if dev_mode or verdict.get('category') == 'SOLVABLE':
            recovery_target = panic_data['high_30d']
            opportunity = find_cornwall_option(
                panic_data['stock_obj'], 
                panic_data['price'], 
                recovery_target,
                dev=dev_mode
            )
            
            if dev_mode:
                if opportunity: st.text("🔹 Math Passed")
                else: st.warning("Dev Mode: Math Step Failed")
            
            min_ratio = 2.0 if dev_mode else 8.0
            
            if opportunity and opportunity['ratio'] > min_ratio:
                found_opportunities.append({
                    "ticker": ticker,
                    "data": panic_data,
                    "verdict": verdict,
                    "option": opportunity
                })
                
                if dev_mode:
                    status.text("✅ Dev Mode: Setup Forced & Displayed.")
                    break
    
    status.empty()
    bar.empty()
    
    if dev_mode or st.session_state.debug_logs:
        with st.expander("🔍 Debug Logs", expanded=True):
            for log in st.session_state.debug_logs:
                st.text(log)
    
    if not found_opportunities:
        st.info("No asymmetric opportunities found.")
    else:
        st.success(f"Found {len(found_opportunities)} Cornwall Setups")
        
        for opp in found_opportunities:
            t = opp['ticker']
            d = opp['data']
            v = opp['verdict']
            o = opp['option']
            
            try:
                raw_date = o['expiration']
                fmt_date = datetime.strptime(raw_date, "%Y-%m-%d").strftime("%B %d %Y")
            except:
                fmt_date = o['expiration']

            with st.container(border=True):
                st.markdown(f"### {t} <span style='font-size:16px; color:#ff4b4b;'>{d['drop_pct']:.1f}% Drop</span>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([2, 4, 2])
                with c1:
                    st.metric("Price", f"${d['price']:.2f}")
                    st.caption(f"Target: ${d['high_30d']:.2f}")
                with c2:
                    cat = v.get('category', 'UNKNOWN')
                    color = "#00C853" if cat == 'SOLVABLE' else "#FFA726"
                    st.markdown(f"**AI:** <span style='color:{color}; font-weight:bold;'>{cat}</span>", unsafe_allow_html=True)
                    st.write(f"_{v.get('reason')}_")
                with c3:
                    st.markdown(f"""
                    <div style='font-size: 14px;'>
                    <b>{o['strike']:.2f} Strike Call</b><br>
                    Expiration: {fmt_date}<br>
                    Estimated Cost: ${o['ask']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("📚 Analyzed Intelligence (Click to verify)"):
                    st.markdown(v.get('sources', 'No sources available.'))
