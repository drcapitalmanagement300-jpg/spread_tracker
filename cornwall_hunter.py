import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import requests
import xml.etree.ElementTree as ET

# --- SILENCE WARNINGS ---
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (New Router V25)</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Test Logic)", value=False, help="Forces detection of 'Panic' to test AI and UI logic.")
    
    st.divider()
    st.write("### 🔌 Connection Debugger")
    
    if st.button("Test Hugging Face Connection"):
        api_key = st.secrets.get("HUGGINGFACE_API_KEY")
        if not api_key:
            st.error("Missing HUGGINGFACE_API_KEY")
        else:
            try:
                # NEW ROUTER URL
                API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {"inputs": "Reply with one word: Connected"}
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    st.success(f"✅ Success: {response.json()[0]['generated_text']}")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

# --- EXPANDED UNIVERSE (Cleaned & De-Duplicated) ---
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
    """Fetches real news from Google News RSS."""
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

# --- PHASE 1: QUANT SCANNER (BULLDOG MODE) ---
def scan_for_panic(ticker, dev=False):
    wait_time = 10 
    
    while True:
        try:
            stock = yf.Ticker(ticker)
            
            if dev:
                return {
                    "ticker": ticker,
                    "price": 100.0,
                    "high_30d": 145.0, 
                    "drop_pct": -30.0,
                    "hv": 85.0,
                    "stock_obj": stock
                }

            hist = stock.history(period="3mo")
            
            if hist.empty: 
                log_debug(f"{ticker}: No history found.")
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
            err_msg = str(e)
            if "Too Many Requests" in err_msg or "Rate limited" in err_msg or "429" in err_msg:
                st.warning(f"⚠️ Rate Limit on {ticker}. Pausing {wait_time}s...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 60)
                continue 
            else:
                log_debug(f"{ticker}: Quant Error {err_msg}")
                return None

# --- PHASE 2: AI CLASSIFIER (NEW ROUTER) ---
def analyze_solvency_hf(ticker, stock_obj, api_key, dev=False):
    try:
        news_text = ""
        source_used = "None"
        
        # 1. Try yfinance News
        try:
            news_items = stock_obj.news
            if news_items:
                headlines = []
                for n in news_items[:5]:
                    t = n.get('title')
                    if t and len(t) > 10: headlines.append(f"- {t}")
                if headlines:
                    news_text = "\n".join(headlines)
                    source_used = "Yahoo Finance"
        except: pass
            
        # 2. Fallback: Google News RSS
        if not news_text:
            log_debug(f"{ticker}: Yahoo empty. Fetching Google News RSS...")
            news_text = get_google_news(ticker)
            if news_text: source_used = "Google News RSS"
            
        if not news_text:
            if dev: 
                news_text = "Stock down on macro fears."
                source_used = "Dev Mock"
            else:
                return {"category": "UNKNOWN", "reason": "No news found."}
        
        # --- NEW URL STRUCTURE ---
        # Note the "/hf-inference/" part which is critical now
        API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        prompt = f"""[INST] You are a financial risk analyst.
        Analyze these headlines for {ticker}:
        {news_text}
        
        Is this a TERMINAL problem (Fraud, Bankruptcy) or SOLVABLE (Earnings Miss, Macro)?
        
        Respond with valid JSON only: {{"category": "TERMINAL" or "SOLVABLE", "reason": "summary"}}
        [/INST]"""
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100, "return_full_text": False}
        }
        
        # Retry logic for AI
        for _ in range(3):
            response = requests.post(API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                break
            time.sleep(2)
        
        if response.status_code != 200:
            log_debug(f"HF Error {response.status_code}: {response.text}")
            return {"category": "ERROR", "reason": "HF API Limit"}
            
        result_text = response.json()[0]['generated_text']
        
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx]
            parsed = json.loads(json_str)
            parsed['sources'] = f"**Source: {source_used}**\n\n{news_text}"
            return parsed
        else:
            return {"category": "ERROR", "reason": "Bad JSON from AI"}
            
    except Exception as e:
        log_debug(f"AI Critical Error: {str(e)}")
        return {"category": "ERROR", "reason": "AI Critical Failure"}

# --- PHASE 3: OPTION MATH ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    try:
        exps = stock_obj.options
        
        if dev:
            return {
                "expiration": "2026-06-18",
                "strike": round(normal_price, 0),
                "ask": 0.50,
                "ratio": 20.0,
                "contract": {"strike": normal_price, "ask": 0.50}
            }
            
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
    
    scan_list = LIQUID_TICKERS[:1] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # POLITE SCANNING
        time.sleep(2.0) 
        
        # 1. QUANT
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Intel gathering...")
        
        # 2. AI (Hugging Face)
        time.sleep(1.0) 
        verdict = analyze_solvency_hf(ticker, panic_data['stock_obj'], api_key, dev=dev_mode)
        
        if not verdict or verdict.get('category') == "ERROR":
            continue

        # 3. MATH
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
                    status.text("✅ Dev Mode: Opportunity found.")
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
