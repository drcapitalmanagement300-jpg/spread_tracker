import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import random
import os
import sys
from scipy.stats import norm
from google.oauth2 import service_account
from googleapiclient.discovery import build
import finnhub

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Alpha Spread Scanner")

# --- SMART FILE LOADING ---
current_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
key_filename = 'service_account.json'

key_path = None
if os.path.exists(os.path.join(current_dir, key_filename)):
    key_path = os.path.join(current_dir, key_filename)
elif os.path.exists(os.path.join(script_dir, key_filename)):
    key_path = os.path.join(script_dir, key_filename)

# --- API KEYS ---
FINNHUB_API_KEY = "d5mgc39r01ql1f2p69c0d5mgc39r01ql1f2p69cg"

# --- CONNECTION ---
@st.cache_resource
def get_drive_service_local():
    if not key_path:
        st.error(f"❌ CRITICAL ERROR: Could not find '{key_filename}'")
        return None
    SCOPES = ['https://www.googleapis.com/auth/drive']
    try:
        creds = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"Key Error: {e}")
        return None

# --- PERSISTENCE ---
def load_from_drive_local(service):
    try:
        FILE_ID = "1CvUDz7NDGaJgyZxQFL9bi5JS2M7nX6K-"
        from googleapiclient.http import MediaIoBaseDownload
        import io, json
        request = service.files().get_media(fileId=FILE_ID)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False: status, done = downloader.next_chunk()
        fh.seek(0)
        return json.loads(fh.read().decode('utf-8'))
    except: return []

def save_to_drive_local(service, data):
    try:
        FILE_ID = "1CvUDz7NDGaJgyZxQFL9bi5JS2M7nX6K-"
        from googleapiclient.http import MediaIoBaseUpload
        import io, json
        json_str = json.dumps(data, indent=2)
        fh = io.BytesIO(json_str.encode('utf-8'))
        media = MediaIoBaseUpload(fh, mimetype='application/json')
        service.files().update(fileId=FILE_ID, media_body=media).execute()
        return True
    except Exception as e: 
        st.error(f"Save Failed: {e}")
        return False

# --- INIT STATE ---
if "init_done" not in st.session_state:
    st.session_state.init_done = True
    st.session_state.scan_results = []
    st.session_state.scan_log = []
    service = get_drive_service_local()
    if service:
        st.session_state.drive_service = service
        st.session_state.trades = load_from_drive_local(service)
        st.toast("✅ Alpha Engine Online")
    else:
        st.session_state.drive_service = None
        st.session_state.trades = []

# --- CONSTANTS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#FFA726"
BG_COLOR = '#0E1117'
GRID_COLOR = '#444444' 
STRIKE_COLOR = '#FF5252'

# --- SECTOR MAPPING ---
SECTOR_DB = {
    "SPY": "Market ETF", "QQQ": "Tech ETF", "IWM": "Small Cap", "DIA": "Dow ETF", 
    "TLT": "Bonds", "GLD": "Gold", "SLV": "Silver", "XLK": "Tech", "XLF": "Financials", 
    "XLE": "Energy", "XLV": "Healthcare", "XLY": "Discretionary", "XLP": "Staples", 
    "XLI": "Industrial", "XLU": "Utilities", "SMH": "Semis", "EEM": "Emerging", 
    "FXI": "China", "ARKK": "Innovation", "BITO": "Crypto",
    "NVDA": "Semis", "AAPL": "Tech", "MSFT": "Software", "AMZN": "Retail", 
    "GOOGL": "Services", "META": "Social", "TSLA": "Auto", "NFLX": "Media",
    "AMD": "Semis", "AVGO": "Semis", "CRM": "Software", "ADBE": "Software", 
    "ORCL": "Software", "CSCO": "Hardware", "INTC": "Semis", "QCOM": "Semis", 
    "TXN": "Semis", "AMAT": "Semis", "MU": "Semis", "LRCX": "Semis",
    "COIN": "Crypto", "MSTR": "Crypto", "PLTR": "Software", "HOOD": "Fintech", 
    "SQ": "Fintech", "SHOP": "E-Comm", "UBER": "Gig", "ABNB": "Travel", 
    "ROKU": "Media", "DKNG": "Gaming", "NET": "Cloud", "SNOW": "Cloud", 
    "PANW": "CyberSec", "CRWD": "CyberSec", "TTD": "AdTech", "ZS": "CyberSec",
    "JPM": "Banks", "BAC": "Banks", "WFC": "Banks", "C": "Banks", 
    "GS": "Capital Mkts", "MS": "Capital Mkts", "V": "Payments", "MA": "Payments", 
    "AXP": "Credit", "PYPL": "Fintech", "BLK": "Asset Mgmt",
    "HD": "Home Imp.", "LOW": "Home Imp.", "COST": "Retail", "WMT": "Retail", 
    "TGT": "Retail", "NKE": "Apparel", "SBUX": "Restaurants", "MCD": "Restaurants", 
    "DIS": "Entertainment", "CMG": "Restaurants", "LULU": "Apparel",
    "CAT": "Machinery", "DE": "Machinery", "BA": "Aerospace", "GE": "Industrial", 
    "LMT": "Defense", "RTX": "Defense", "XOM": "Oil", "CVX": "Oil", 
    "COP": "Oil", "SLB": "Services",
    "LLY": "Pharma", "UNH": "Insurance", "JNJ": "Pharma", "PFE": "Pharma", 
    "MRK": "Pharma", "ABBV": "Pharma", "BMY": "Pharma", "AMGN": "Biotech"
}
LIQUID_TICKERS = list(SECTOR_DB.keys())
ETFS = [k for k, v in SECTOR_DB.items() if "ETF" in v]

# --- BLACK-SCHOLES ENGINE ---
def black_scholes_delta(S, K, T, r, sigma, option_type="put"):
    try:
        if T <= 0 or sigma <= 0: return -0.5
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "call": return norm.cdf(d1)
        else: return norm.cdf(d1) - 1
    except: return -0.5 

# --- VIX CHECKER ---
def get_market_vix():
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1d")
        if hist.empty: return 15.0 # Fallback
        return hist['Close'].iloc[-1]
    except: return 15.0

# --- EARNINGS CHECKER ---
@st.cache_resource
def get_finnhub_client():
    if not FINNHUB_API_KEY: return None
    return finnhub.Client(api_key=FINNHUB_API_KEY)

def check_earnings_safety(ticker, expiration_date_str):
    if ticker in ETFS: return True, "ETF"
    client = get_finnhub_client()
    if not client: return True, "No API"
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        earnings = client.earnings_calendar(_from=today_str, to=expiration_date_str, symbol=ticker)
        if earnings and 'earningsCalendar' in earnings:
            events = earnings['earningsCalendar']
            if len(events) > 0: return False, events[0]['date']
        return True, "Safe"
    except: return True, "Check"

# --- AI SCORING ENGINE ---
def calculate_alpha_score(data, spread_metrics, market_vix):
    score = 0
    reasons = []
    
    # 1. Trend Quality (30 Pts)
    if data['price'] > data['sma_200']:
        score += 15
        if data['price'] < data['sma_20']: 
            score += 15
            reasons.append("Perfect Dip")
        else:
            reasons.append("Trend Up")
    else:
        reasons.append("Trend Weak")

    # 2. Volatility Edge (30 Pts)
    vrp_ratio = spread_metrics['iv'] / spread_metrics['hv'] if spread_metrics['hv'] > 0 else 1.0
    if vrp_ratio > 1.5: 
        score += 30
        reasons.append("High VRP")
    elif vrp_ratio > 1.2: 
        score += 20
        reasons.append("Good VRP")
    
    # 3. Macro Context (VIX)
    if market_vix > 20:
        score += 10
        reasons.append("High VIX")
    elif market_vix < 12:
        score -= 10 # Penalty for low premiums
        
    # 4. Entry/RSI (20 Pts)
    if data['rsi'] < 40: 
        score += 20
        reasons.append("Oversold")
    elif data['rsi'] < 50:
        score += 10
        
    return min(score, 100), reasons

# --- DATA PROCESSOR ---
def process_market_structure(ticker_obj, ticker):
    try:
        hist = ticker_obj.history(period="1y", auto_adjust=True)
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252)
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            "price": current_price,
            "change_pct": ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
            "sma_20": hist['SMA_20'].iloc[-1],
            "sma_50": hist['SMA_50'].iloc[-1],
            "sma_200": hist['SMA_200'].iloc[-1],
            "hv": hist['HV'].iloc[-1] * 100,
            "rsi": rsi.iloc[-1],
            "sector": SECTOR_DB.get(ticker, "Stock"),
            "hist": hist
        }
    except: return None

# --- SCANNER LOGIC (STRICT WIDTH) ---
def find_alpha_setup(ticker, stock_obj, data, width_target, min_roi, dev_mode):
    try:
        if not dev_mode and data['price'] < data['sma_200']:
            return None, "Bearish Trend (<200 SMA)"

        try: exps = stock_obj.options
        except: return None, "No Options"
        
        valid_exps, monthly_exps = [], []
        now = datetime.now()
        
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d")
                days = (edate - now).days
                if 25 <= days <= 60:
                    valid_exps.append(e)
                    if 15 <= edate.day <= 21: monthly_exps.append(e)
            except: pass
        
        if not valid_exps: return None, "No Valid Expiry"
        best_exp = monthly_exps[0] if monthly_exps else valid_exps[0]
        
        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        if puts.empty: return None, "No Puts"
        
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - now).days
        T, r = dte / 365.0, 0.045
        
        atm_idx = (puts['strike'] - data['price']).abs().argsort()[:1]
        if atm_idx.empty: return None, "Data Err"
        atm_iv = puts.iloc[atm_idx]['impliedVolatility'].values[0]
        if atm_iv == 0: atm_iv = max(data['hv']/100, 0.2)

        puts['delta'] = puts.apply(lambda row: black_scholes_delta(
            data['price'], row['strike'], T, r, 
            row['impliedVolatility'] if row['impliedVolatility'] > 0 else atm_iv
        ), axis=1)

        # Wide Delta Search (18-35)
        candidates = puts[(puts['delta'] > -0.35) & (puts['delta'] < -0.18)]
        if candidates.empty: return None, "No Strikes in Delta Range"

        best_spread = None
        rejection_reason = "Low ROI"

        for _, short_leg in candidates.sort_values('strike', ascending=False).iterrows():
            short_strike = short_leg['strike']
            
            min_oi = 500 if ticker in ETFS else 100
            if not dev_mode and (short_leg.get('openInterest', 0) < min_oi): continue

            # STRICT WIDTH LOGIC: ONLY try the specific target (e.g. $5)
            long_target = short_strike - width_target
            long_leg_idx = (puts['strike'] - long_target).abs().argsort()[:1]
            if long_leg_idx.empty: continue
            long_leg = puts.iloc[long_leg_idx].iloc[0]
            
            # Verify exact width match (within $0.50 tolerance for strikes)
            actual_width = short_strike - long_leg['strike']
            if abs(actual_width - width_target) > 0.5: continue 

            credit = short_leg['bid'] - long_leg['ask']
            risk = actual_width - credit
            
            if risk <= 0 or credit <= 0: continue
            
            roi = (credit / risk) * 100
            
            if roi >= (min_roi if not dev_mode else 5.0):
                is_safe, msg = check_earnings_safety(ticker, best_exp)
                if not is_safe and not dev_mode: 
                    rejection_reason = f"Earnings ({msg})"
                    break
                
                exp_str = datetime.strptime(best_exp, "%Y-%m-%d").strftime("%b %d, %Y")
                
                best_spread = {
                    "ticker": ticker, "expiration_raw": best_exp, "expiration": exp_str,
                    "dte": dte, "short": short_strike, "long": long_leg['strike'],
                    "credit": credit, "risk": risk, "roi": roi, "delta": short_leg['delta'],
                    "iv": short_leg['impliedVolatility'] * 100, "hv": data['hv']
                }
                break
        
        return (best_spread, None) if best_spread else (None, rejection_reason)

    except Exception as e: return None, f"Error: {e}"

# --- PLOTTING ---
def plot_sparkline_cone(hist, short_strike, long_strike, current_price, iv, dte):
    fig, ax = plt.subplots(figsize=(4, 1.3)) 
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    try:
        last_60 = hist.tail(60).copy()
        dates = last_60.index
        bar_colors = np.where(last_60['Close'] >= last_60['Open'], SUCCESS_COLOR, WARNING_COLOR)
        heights = last_60['Close'] - last_60['Open']
        bottoms = last_60['Open']
        
        ax.bar(dates, heights, bottom=bottoms, color=bar_colors, width=0.8, align='center', alpha=0.9)
        
        if iv > 0 and dte > 0:
            last_date = dates[-1]
            future_days = np.arange(1, dte + 5)
            future_dates = [last_date + timedelta(days=int(d)) for d in future_days]
            std_move = current_price * (iv / 100.0) * np.sqrt(future_days / 365.0)
            upper_cone = current_price + std_move
            lower_cone = current_price - std_move
            ax.plot(future_dates, upper_cone, color=SUCCESS_COLOR, linestyle=':', linewidth=1, alpha=0.6)
            ax.plot(future_dates, lower_cone, color=SUCCESS_COLOR, linestyle=':', linewidth=1, alpha=0.6)
            ax.fill_between(future_dates, lower_cone, upper_cone, color=SUCCESS_COLOR, alpha=0.1)
        
        ax.axhline(y=short_strike, color=STRIKE_COLOR, linestyle='-', linewidth=1, alpha=0.9)
        ax.axhline(y=long_strike, color=STRIKE_COLOR, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.grid(True, which='major', linestyle=':', color=GRID_COLOR, alpha=0.3)
    except: pass
    
    ax.axis('off') 
    plt.tight_layout(pad=0.1)
    return fig

# --- HEADER ---
st.markdown("""
<div style='text-align: left; padding-top: 10px;'>
    <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Alpha Spread Scanner <span style='font-size:16px; color:#00C853; vertical-align:middle;'>[AI Enabled]</span></h1>
    <p style='margin-top: 0px; font-size: 14px; color: #888;'>
        Filtering for: <strong style='color:#00C853'>Trend Stack</strong> • 
        <strong style='color:#29B6F6'>Vol Risk Premium</strong> • 
        <strong style='color:#FFA726'>Skew Edge</strong>
    </p>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Alpha Settings")
    
    # LIVE VIX CHECK
    current_vix = get_market_vix()
    
    # VIX Logic
    if current_vix < 13:
        vix_status = "Low Edge (Cheap)"
        vix_color = "#AAA"
    elif 13 <= current_vix <= 20:
        vix_status = "Optimal Income Zone"
        vix_color = SUCCESS_COLOR
    elif 20 < current_vix <= 30:
        vix_status = "High Premium (Juicy)"
        vix_color = "#FFA726"
    else:
        vix_status = "Extreme Fear (Risky)"
        vix_color = "#FF5252"

    st.markdown(f"""
    <div style='border:1px solid {vix_color}; padding:10px; border-radius:5px; text-align:center; margin-bottom:15px;'>
        <div style='font-size:12px; color:#AAA; text-transform:uppercase;'>Market Volatility (VIX)</div>
        <div style='font-size:24px; font-weight:bold; color:{vix_color};'>{current_vix:.2f}</div>
        <div style='font-size:11px; color:{vix_color}; margin-top:2px;'>({vix_status})</div>
    </div>
    """, unsafe_allow_html=True)
    
    dev_mode = st.checkbox("Dev Mode (Bypass Filters)", value=False)
    width_target = st.slider("Spread Width ($)", 1, 10, 5, 1) # Strict Width
    min_roi = st.slider("Min ROI %", 5, 30, 10, 1)
    if st.button("Reset Scanner"):
        st.session_state.scan_results = []
        st.session_state.scan_log = []
        st.rerun()

# --- SCANNER ---
if st.button("Run Alpha Scan"):
    st.session_state.scan_results = []
    st.session_state.scan_log = []
    
    if not st.session_state.drive_service:
        st.error("❌ Drive Missing")
        st.stop()
        
    progress = st.progress(0)
    status = st.empty()
    log_box = st.empty()
    
    current_vix = get_market_vix() # Fetch once for the run
    
    total = len(LIQUID_TICKERS)
    
    for i, ticker in enumerate(LIQUID_TICKERS):
        progress.progress((i + 1) / total)
        status.text(f"Scanning {ticker}...")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = process_market_structure(ticker_obj, ticker)
            
            if not data:
                st.session_state.scan_log.append(f"[{ticker}] No Data")
                continue
                
            spread, reason = find_alpha_setup(
                ticker, ticker_obj, data, width_target, min_roi, dev_mode
            )
            
            if spread:
                ai_score, reasons = calculate_alpha_score(data, spread, current_vix)
                
                if ai_score >= 50 or dev_mode:
                    st.session_state.scan_log.append(f"[{ticker}] 🚀 MATCH | Score: {ai_score}")
                    st.session_state.scan_results.append({
                        "ticker": ticker, "data": data, "spread": spread, 
                        "score": ai_score, "reasons": reasons, "vix": current_vix
                    })
            else:
                st.session_state.scan_log.append(f"[{ticker}] {reason}")
                
        except Exception as e:
            st.session_state.scan_log.append(f"[{ticker}] Error: {str(e)}")
            
        log_box.code("\n".join(st.session_state.scan_log[-8:]))
        time.sleep(0.1)

    status.text("Alpha Scan Complete")

# --- RESULTS GRID ---
if st.session_state.scan_results:
    sorted_results = sorted(st.session_state.scan_results, key=lambda x: x['score'], reverse=True)
    cols = st.columns(3)
    
    for i, res in enumerate(sorted_results):
        t = res['ticker']
        d = res['data']
        s = res['spread']
        score = res['score']
        vix = res.get('vix', 15)
        
        with cols[i % 3]:
            with st.container(border=True):
                # Header
                pill_color = "price-pill-green" if d['change_pct'] >= 0 else "price-pill-red"
                
                if score >= 80: badge = "💎 ALPHA SETUP"
                elif score >= 65: badge = "⭐ PRIME EDGE"
                else: badge = "✔️ VIABLE"
                
                reasons_text = " + ".join(res['reasons']) if res['reasons'] else "Standard"
                
                # Vix Warning Label
                vix_warning = ""
                if vix < 13: vix_warning = "<span style='color:#FF5252; font-weight:bold; font-size:11px;'>⚠️ LOW VIX</span>"
                elif vix > 30: vix_warning = "<span style='color:#FF5252; font-weight:bold; font-size:11px;'>⚠️ HIGH VIX</span>"

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 22px; font-weight: 900; color: white;">{t} <span style="font-size: 12px; color: #888; font-weight: normal;">• {d['sector']}</span></div>
                        <div style="margin-top: 4px;"><span class="{pill_color}">${d['price']:.2f} ({d['change_pct']:.2f}%)</span></div>
                    </div>
                    <div style="text-align: right;">
                        <div class="strategy-badge">{badge}</div>
                        <div style="font-size: 20px; font-weight: 900; color: #d4ac0d; margin-top: 4px;">{score}</div>
                    </div>
                </div>
                <div style="margin-top: 5px;">
                    <span style="font-size: 11px; color: #00E676;">⚡ {reasons_text}</span>
                    <span style="float:right;">{vix_warning}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-label">Strikes</div>
                    <div class="metric-value">${s['short']:.0f} / ${s['long']:.0f}</div>
                    <div style="height: 8px;"></div>
                    <div style="display: flex; gap: 15px;">
                        <div>
                            <div class="metric-label">Credit</div>
                            <div class="metric-value" style="color:{SUCCESS_COLOR}">${s['credit']:.2f}</div>
                        </div>
                        <div>
                            <div class="metric-label">Max Risk</div>
                            <div class="metric-value" style="color:#FF5252">${s['risk']:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-label">Expiry</div>
                    <div class="metric-value">{s['expiration']} <span style="color:#888; font-size:12px;">({s['dte']}d)</span></div>
                    <div style="height: 8px;"></div>
                    <div class="metric-label">Volatility</div>
                    <div><span style="color:#29B6F6; font-weight:bold;">{s['iv']:.1f}%</span> <span style="color:#666; font-size:11px;">(vs {s['hv']:.1f})</span></div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.pyplot(plot_sparkline_cone(d['hist'], s['short'], s['long'], d['price'], s['iv'], s['dte']), use_container_width=True)
                
                st.markdown(f"""<div class="roc-box"><span style="font-size:11px; color: #00c864; text-transform: uppercase;">Return on Capital</span><br><span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.1f}%</span></div>""", unsafe_allow_html=True)
                
                st.write("") 
                col_qty, col_btn = st.columns([1, 2])
                with col_qty:
                    qty = st.number_input("Qty", 1, 100, 1, key=f"q{i}", label_visibility="collapsed")
                with col_btn:
                    add_key = f"add_{t}_{i}"
                    btn_key = f"btn_state_{t}_{i}"
                    if btn_key not in st.session_state: st.session_state[btn_key] = False

                    if not st.session_state[btn_key]:
                        if st.button(f"Add Trade", key=add_key, use_container_width=True):
                            new_trade = {
                                "id": f"{t}-{s['short']}-{s['expiration_raw']}",
                                "ticker": t, "contracts": qty, "short_strike": s['short'], "long_strike": s['long'],
                                "expiration": s['expiration_raw'], "credit": s['credit'],
                                "entry_date": datetime.now().date().isoformat(), "pnl_history": []
                            }
                            st.session_state.trades.append(new_trade)
                            save_to_drive_local(st.session_state.drive_service, st.session_state.trades)
                            st.session_state[btn_key] = True
                            st.rerun()
                    else:
                        st.success(f"✅ Added {qty} {t}")

# --- CSS ---
st.markdown("""<style>
.metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
.metric-value { font-size: 16px; font-weight: 700; color: #FFF; }
.price-pill-red { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
.price-pill-green { background-color: rgba(0, 200, 100, 0.15); color: #00c864; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
.strategy-badge { color: #d4ac0d; padding: 2px 8px; font-size: 12px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; text-align: right; }
.roc-box { background-color: rgba(0, 255, 127, 0.05); border: 1px solid rgba(0, 255, 127, 0.2); border-radius: 6px; padding: 8px; text-align: center; margin-top: 12px; }
</style>""", unsafe_allow_html=True)