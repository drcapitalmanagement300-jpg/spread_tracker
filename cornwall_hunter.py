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
# These are stocks that actually crash 30-50% and recover.
LIQUID_TICKERS = [
    # --- HIGH BETA / GROWTH ---
    "PLTR", "SOFI", "HOOD", "DKNG", "ROKU", "SHOP", "SQ", "AFRM", "UPST", "CVNA",
    "NET", "DDOG", "SNOW", "U", "RBLX", "COIN", "MSTR", "MARA", "RIOT", "CLSK",
    
    # --- VOLATILE TECH & AI ---
    "AI", "IONQ", "PLUG", "FCEL", "JOBY", "ACHR", "SPCE", "ASTS", "LUNR",
    
    # --- BIOTECH (The Kings of Binary Events) ---
    "XBI", "LABU", "MRNA", "BNTX", "CRSP", "NTLA", "EDIT", "BEAM",
    
    # --- MEME / RETAIL FAVORITES ---
    "GME", "AMC", "CHWY", "RDDT", "DJT",
    
    # --- DISASTER RECOVERY (Legacy) ---
    "BA", "INTC", "WBA", "PARA", "LULU", "NKE", "SBUX", "PYPL",
    
    # --- INDEXES (For Reference) ---
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
        # Fetch 3mo to ensure we have enough data
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        last_30 = hist.tail(30)
        high_30d = last_30['High'].max()
        
        if high_30d <= 0: return None
        
        drop_pct = ((current_price - high_30d) / high_30d) * 100
        
        # Calculate Volatility
        hist['Returns'] = hist['Close'].pct_change()
        current_hv = hist['Returns'].tail(30).std() * np.sqrt(252) * 100
        
        # DEV OVERRIDE
        if dev:
            if drop_pct > -5.0:
                return {
                    "ticker": ticker,
                    "price": current_price,
                    "high_30d": current_price * 1.25, 
                    "drop_pct": -20.0, 
                    "hv": 85.0,        
                    "stock_obj": stock
                }
        
        # PRODUCTION LOGIC
        # We want >15% drop and >40 HV (Mid-caps need higher vol to be interesting)
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
        
        # Find LEAPS (options > 250 days out)
        leaps = [e for e in exps if (datetime.strptime(e, "%Y-%m-%d") - datetime.now()).days > 250]
        if not leaps: return None
        
        target_exp = leaps[0] 
        chain = stock_obj.option_chain(target_exp)
        calls = chain.calls
        
        # Target Strike = The "Normal" Price
        target_strike = normal_price
        
        # Find strikes within 5% of the target
        candidates = calls[(calls['strike'] >= target_strike * 0.95) & (calls['strike'] <= target_strike * 1.05)]
        
        if candidates.empty and dev:
             # Widen search for dev mode
             candidates = calls[(calls['strike'] >= target_strike * 0.80) & (calls['strike'] <= target_strike * 1.20)]

        if candidates.empty: return None
        
        # Pick the most liquid one
        option = candidates.sort_values('volume', ascending=False).iloc[0]
        ask_price = option['ask']
        
        if ask_price <= 0:
            if dev: ask_price = 0.50
            else: return None
        
        # Cornwall Ratio: (Normal Price - Strike) / Premium
        # Since Strike ~= Normal Price, we are basically buying ATM relative to "Normal"
        # But relative to "Current", it is Deep OTM.
        
        # For simplicity in this scanner, we look for asymmetric payout potential
        # If stock returns to normal, the option is worth (Normal - Strike).
        # Wait, if Strike = Normal, Intrinsic is 0. 
        # CORRECTION: We want Strike to be slightly BELOW Normal, or we calculate payoff based on OVERSHOOT.
        
        # Cornwall Strategy often buys OTM relative to CURRENT, but ITM relative to RECOVERY.
        # Let's adjust: We buy the strike at the HALFWAY point between Current and Normal.
        # Or we stick to the user's prompt: "find mispriced OTMs"
        
        projected_intrinsic = normal_price - option['strike']
        
        if projected_intrinsic <= 0:
            # If strike is above normal, we need it to go HIGHER than normal to profit
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
    
    # 1. Dev Mode: Use tiny list. Production: Use full list.
    scan_list = LIQUID_TICKERS[:3] if dev_mode else LIQUID_TICKERS
    
    # 2. Rate Limit Logic (PERSISTENT)
    RATE_LIMIT_DELAY = 4.1 
    
    # Initialize session state for rate limiting if not present
    if "last_gemini_call" not in st.session_state:
        st.session_state.last_gemini_call = 0

    for i, ticker in enumerate(scan_list):
        status.text(f"Scanning {ticker}...")
        bar.progress((i+1) / len(scan_list))
        
        # --- QUANT (Fast, Local) ---
        panic_data = scan_for_panic(ticker, dev=dev_mode)
        if not panic_data: continue
        
        status.text(f"💥 Panic: {ticker}. Asking Gemini...")
        
        # --- SMART THROTTLE (PERSISTENT) ---
        # Calculate time since the GLOBAL last call
        elapsed = time.time() - st.session_state.last_gemini_call
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            status.text(f"⏳ Throttling for {sleep_time:.1f}s (Gemini Free Tier)...")
            time.sleep(sleep_time)
        
        # --- AI CALL ---
        verdict = analyze_solvency_gemini(ticker, panic_data['stock_obj'], dev=dev_mode)
        
        # Update Global Timer
        st.session_state.last_gemini_call = time.time()
        
        # Fail safe
        if verdict.get('category') == "RATE_LIMIT":
            st.warning("Rate limit hit. Cooling down 60s...")
            time.sleep(60)
            verdict = analyze_solvency_gemini(ticker, panic_data['stock_obj'], dev=dev_mode)
            st.session_state.last_gemini_call = time.time()

        # --- RESULTS ---
        if verdict.get('category') == 'SOLVABLE' or dev_mode:
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
                
                # DEV MODE STOPPER
                if dev_mode:
                    status.text("Dev Mode: Result found. Stopping early.")
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
