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

# --- OPENAI STANDARD LIBRARY ---
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
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Solvable Problem vs. Terminal Risk Classifier (Google News Intel)</p>
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
                with st.spinner("Pinging Free Model..."):
                    completion = client.chat.completions.create(
                        model="google/gemini-2.0-flash-lite-preview-02-05:free",
                        messages=[{"role": "user", "content": "Reply with one word: Connected"}]
                    )
                st.success(f"✅ Success: {completion.choices[0].message.content}")
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

# --- HELPER: GOOGLE NEWS RSS FETCHER ---
def get_google_news(ticker):
    """Fetches real news from Google News RSS when Yahoo fails."""
    try:
        # Search for Ticker + Stock to avoid generic results
        query = f"{ticker} stock news"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return ""
            
        root = ET.fromstring(response.content)
        headlines = []
        
        # Parse XML for titles
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text
            pubDate = item.find('pubDate').text
            headlines.append(f"- {title} ({pubDate})")
            
        return "\n".join(headlines)
    except Exception as e:
        log_debug(f"Google News Fetch Failed for {ticker}: {e}")
        return ""

# --- PHASE 1: QUANT SCANNER ---
def scan_for_panic(ticker, dev=False):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: 
            log_debug(f"{ticker}: No history found.")
            return None
        
        current_price = hist['Close'].iloc[-1]
        
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

# --- PHASE 2: AI CLASSIFIER (GOOGLE NEWS BACKED) ---
def analyze_solvency_openrouter(ticker, stock_obj, client, dev=False):
    try:
        news_text = ""
        source_used = "None"
        
        # 1. Try yfinance News First
        try:
            news_items = stock_obj.news
            if news_items:
                headlines = []
                for n in news_items[:5]:
                    t = n.get('title')
                    if t and len(t) > 10:
                        headlines.append(f"- {t}")
                if headlines:
                    news_text = "\n".join(headlines)
                    source_used = "Yahoo Finance"
        except: pass
            
        # 2. Fallback: Google News RSS (Reliable)
        if not news_text:
            log_debug(f"{ticker}: Yahoo empty. Fetching Google News RSS...")
            news_text = get_google_news(ticker)
            if news_text:
                source_used = "Google News RSS"
            
        # 3. If STILL no news
        if not news_text:
            if dev: 
                # DEV MODE: Inject fake bad news if we can't find real news
                news_text = "- Stock crashes 20% on earnings miss due to temporary supply chain issue.\n- Analyst downgrades but maintains long term buy.\n- CEO says headwinds are transient."
                source_used = "Dev Mode Simulation"
            else:
                return {
                    "category": "UNKNOWN", 
                    "reason": "No news found via Yahoo or Google News.",
                    "sources": "No articles found."
                }
        
        FREE_MODELS = [
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "mistralai/mistral-7b-instruct:free"
        ]
        
        prompt = f"""
        Analyze the recent news for {ticker} causing the price drop. 
        Headlines:
        {news_text}
        
        Classify the crisis into one of two categories:
        1. TERMINAL: Fraud, Bankruptcy, Criminal Indictment, Accounting Scandal, Core Business Obsolete.
        2. SOLVABLE: Earnings Miss, CEO Fired, Regulatory Fine, Lawsuit, Product Recall, Macro Fear.
        
        If the information is vague, assume SOLVABLE (Macro Volatility).
        
        Return JSON format: {{"category": "TERMINAL" or "SOLVABLE", "reason": "Short summary (max 15 words)"}}
        """
        
        last_error = ""
        
        for model_id in FREE_MODELS:
            try:
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
                if "<think>" in content: 
                    content = content.split("</think>")[-1].strip()
                    
                content = content.replace("```json", "").replace("```", "").strip()
                
                parsed = json.loads(content)
                if isinstance(parsed, list): parsed = parsed[0] if parsed else {}
                if not isinstance(parsed, dict): continue
                
                parsed['sources'] = f"**Source: {source_used}**\n\n{news_text}"
                return parsed
                
            except Exception as e:
                last_error = str(e)
                continue
        
        log_debug(f"All AI models failed. Last error: {last_error}")
        return {"category": "ERROR", "reason": "AI Models Offline", "sources": "AI Connection Failed"}
        
    except Exception as e:
        log_debug(f"AI Critical Error: {str(e)}")
        return {"category": "ERROR", "reason": "AI Critical Failure", "sources": str(e)}

# --- PHASE 3: OPTION MATH (TRUE LEAPS) ---
def find_cornwall_option(stock_obj, current_price, normal_price, dev=False):
    try:
        exps = stock_obj.options
        
        if dev:
            return {
                "expiration": "2026-06-18",
                "strike": round(normal_price, 0),
                "ask": 0.50,
                "ratio": 20.0,
                "contract": {"strike": normal_price, "ask": 0.50, "volume": 1000}
            }
            
        if not exps: return None
        
        # --- STRICT LEAPS: Must be > 360 Days ---
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 360]
        
        if not leaps: 
            log_debug("No LEAPS > 360d found.")
            return None
        
        target_exp = leaps[-1] 
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
    
    scan_list = LIQUID_TICKERS[:1] if dev_mode else LIQUID_TICKERS
    
    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # 1. QUANT
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Intel gathering...")
        
        # 2. AI (MULTI-SOURCE)
        time.sleep(0.5) 
        verdict = analyze_solvency_openrouter(ticker, panic_data['stock_obj'], client, dev=dev_mode)
        
        if not verdict or not isinstance(verdict, dict):
            continue
            
        if verdict.get('category') == "ERROR":
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
            
            # Format Date nicely
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
                    # Clean UI, no green, no bold
                    st.markdown(f"""
                    <div style='font-size: 14px;'>
                    <b>{o['strike']:.2f} Strike Call</b><br>
                    Expiration: {fmt_date}<br>
                    Estimated Cost: ${o['ask']:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("📚 Analyzed Intelligence (Click to verify)"):
                    st.markdown(v.get('sources', 'No sources available.'))
