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

# --- OPENAI STANDARD LIBRARY (For OpenRouter) ---
from openai import OpenAI

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
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (OpenRouter Multi-Model)</p>
    </div>""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Hunter Settings")
    dev_mode = st.checkbox("🛠 Dev Mode (Test Logic)", value=False, help="Forces detection of 'Panic' to test AI and UI logic.")
    
    st.divider()
    st.write("### 🔌 Connection Debugger")
    
    if st.button("Test OpenRouter Connection"):
        api_key = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key:
            st.error("Missing OPENROUTER_API_KEY")
        else:
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                # Try the most reliable free model first
                with st.spinner("Pinging DeepSeek R1 (Free)..."):
                    completion = client.chat.completions.create(
                        model="deepseek/deepseek-r1:free",
                        messages=[{"role": "user", "content": "Reply with one word: Connected"}]
                    )
                st.success(f"✅ Success: {completion.choices[0].message.content}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

# --- EXPANDED UNIVERSE ---
LIQUID_TICKERS = [
    "PLTR", "SOFI", "HOOD", "DKNG", "ROKU", "SHOP", "SQ", "AFRM", "UPST", "CVNA",
    "NET", "DDOG", "SNOW", "U", "RBLX", "COIN", "MSTR", "MARA", "RIOT", "CLSK",
    "AI", "IONQ", "PLUG", "FCEL", "JOBY", "ACHR", "SPCE", "ASTS", "LUNR",
    "XBI", "LABU", "MRNA", "BNTX", "CRSP", "NTLA", "EDIT", "BEAM",
    "GME", "AMC", "CHWY", "RDDT", "DJT",
    "BA", "INTC", "WBA", "PARA", "LULU", "NKE", "SBUX", "PYPL",
    "SPY", "QQQ", "IWM", "ARKK", "KRE"
]

# --- SETUP CLIENT ---
def get_ai_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

# --- DEBUG LOGGER ---
def log_debug(msg):
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {msg}")

# --- PHASE 1: QUANT SCANNER ---
def scan_for_panic(ticker, dev=False):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: 
            log_debug(f"{ticker}: No history found.")
            return None
        
        current_price = hist['Close'].iloc[-1]
        
        # --- DEV MODE FORCE PASS ---
        if dev:
            return {
                "ticker": ticker,
                "price": current_price,
                "high_30d": current_price * 1.40, 
                "drop_pct": -30.0,
                "hv": 85.0,
                "stock_obj": stock
            }

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
        log_debug(f"{ticker}: Quant Error {str(e)}")
        return None

# --- PHASE 2: AI CLASSIFIER (MULTI-MODEL FALLBACK) ---
def analyze_solvency_openrouter(ticker, stock_obj, client, dev=False):
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
        
        # --- ROBUST FREE MODEL LIST ---
        # If one fails (404/429), it automatically tries the next.
        FREE_MODELS = [
            "deepseek/deepseek-r1:free",          # DeepSeek R1 (Very smart, often free)
            "google/gemini-2.0-flash-exp:free",   # Gemini 2.0 Flash Exp (Google's free tier on OR)
            "meta-llama/llama-3.2-3b-instruct:free", # Llama 3 (Reliable fallback)
            "mistralai/mistral-7b-instruct:free", # Mistral 7B (Solid fallback)
            "google/gemini-2.0-flash-lite-preview-02-05:free" # New Lite model
        ]
        
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear.
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary (max 15 words)"}}
        """
        
        last_error = ""
        
        for model_id in FREE_MODELS:
            try:
                # log_debug(f"Trying model: {model_id}...") # Optional verbose logging
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a financial risk analyst. Output only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    extra_headers={
                        "HTTP-Referer": "https://streamlit.app",
                        "X-Title": "CornwallHunter"
                    }
                )
                
                content = completion.choices[0].message.content
                # Clean response (DeepSeek sometimes adds <think> tags, remove them if needed, mostly handled by JSON parser)
                if "<think>" in content:
                    content = content.split("</think>")[-1].strip()
                    
                content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
                
            except Exception as e:
                # If this model failed, loop to the next one
                last_error = str(e)
                continue
        
        # If ALL models failed
        log_debug(f"All AI models failed. Last error: {last_error}")
        return {"category": "ERROR", "reason": "All Free Models Busy/Offline"}
        
    except Exception as e:
        log_debug(f"AI Critical Error: {str(e)}")
        return {"category": "ERROR", "reason": "AI Critical Failure"}

# --- PHASE 3: OPTION MATH ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    try:
        exps = stock_obj.options
        
        # --- DEV MODE MOCK ---
        if dev:
            return {
                "expiration": "2025-06-20",
                "strike": round(normal_price, 0),
                "ask": 0.50,
                "ratio": 20.0,
                "contract": {"strike": normal_price, "ask": 0.50, "volume": 1000}
            }
            
        if not exps: return None
        
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 250]
        if not leaps: 
            log_debug("No LEAPS > 250d.")
            return None
        
        target_exp = leaps[0] 
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        
        target_strike = normal_price
        
        candidates = calls[(calls['strike'] >= target_strike * 0.80) & (calls['strike'] <= target_strike * 1.20)].copy()
        
        if candidates.empty: 
            log_debug("No strikes near target.")
            return None

        def get_valid_price(row):
            ask = row.get('ask', 0)
            bid = row.get('bid', 0)
            last = row.get('lastPrice', 0)
            if ask > 0: return ask
            if bid > 0 and last > 0: return (bid + last) / 2
            if last > 0: return last
            return 0

        candidates['valid_price'] = candidates.apply(get_valid_price, axis=1)
        candidates['liquidity'] = candidates['volume'].fillna(0) + candidates['openInterest'].fillna(0)
        candidates = candidates[candidates['valid_price'] > 0]
        
        if candidates.empty: return None
            
        option = candidates.sort_values('liquidity', ascending=False).iloc[0]
        
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
    except Exception as e:
        log_debug(f"Option Error: {str(e)}")
        return None

# --- MAIN EXECUTION ---

if st.button(f"Start Hunt {'(Dev Mode)' if dev_mode else ''}"):
    client = get_ai_client()
    if not client:
        st.error("Missing OPENROUTER_API_KEY in secrets.toml")
        st.stop()
    
    status = st.empty()
    bar = st.progress(0)
    st.session_state.debug_logs = [] 
    found_opportunities = []
    
    # DEV MODE SPEED
    scan_list = LIQUID_TICKERS[:1] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # 1. QUANT
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Analyzing news...")
        
        # 2. AI (OPENROUTER MULTI-MODEL)
        # Add a tiny sleep to be nice to the free tier
        time.sleep(1) 
        verdict = analyze_solvency_openrouter(ticker, panic_data['stock_obj'], client, dev=dev_mode)
        
        if verdict.get('category') == "ERROR":
            log_debug(f"{ticker}: Skipped (AI Error - All models failed)")
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
                    st.metric("Payout", f"{o['ratio']:.1f}x")
                    st.markdown(f"**{o['expiration']} ${o['strike']:.0f} C**")
                    st.markdown(f"**Cost:** ${o['ask']:.2f}")
