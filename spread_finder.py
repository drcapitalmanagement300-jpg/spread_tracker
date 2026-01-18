import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
from datetime import datetime, timedelta
import finnhub
import time
import random
import requests

# Import persistence
from persistence import (
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout 
)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Finder")

# --- API KEYS ---
try:
    FINNHUB_API_KEY = st.secrets["general"]["FINNHUB_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("FINNHUB_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

# --- CONSTANTS & COLORS ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
GRID_COLOR = '#444444'
STRIKE_COLOR = '#FF5252'

# --- SECTOR MAP ---
SECTOR_MAP = {
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF", "DIA": "Dow Jones ETF",
    "GLD": "Gold Trust", "SLV": "Silver Trust", "TLT": "20+ Yr Treasury Bond", "XLK": "Technology ETF",
    "XLF": "Financials ETF", "XLE": "Energy ETF", "XLV": "Health Care ETF", "XLI": "Industrials ETF",
    "XLP": "Cons. Staples ETF", "XLU": "Utilities ETF", "XLY": "Cons. Discret. ETF", "SMH": "Semiconductor ETF",
    "ARKK": "Innovation ETF", "KRE": "Regional Banking ETF", "XBI": "Biotech ETF", "GDX": "Gold Miners ETF",
    "EEM": "Emerging Markets", "FXI": "China Large-Cap", "EWZ": "Brazil Capped ETF", "HYG": "High Yield Bond",
    "LQD": "Inv Grade Corp Bond", "UVXY": "Short-Term Futures", "BITO": "Bitcoin Strategy", "USO": "Oil Fund",
    "UNG": "Natural Gas Fund", "TQQQ": "ProShares UltraPro QQQ", "SQQQ": "ProShares UltraPro Short",
    "SOXL": "Direxion Daily Semi Bull", "SOXS": "Direxion Daily Semi Bear",
    "NVDA": "Semiconductors", "TSLA": "Auto Manufacturers", "AAPL": "Consumer Electronics", "MSFT": "Software - Infra",
    "AMD": "Semiconductors", "AMZN": "Internet Retail", "META": "Internet Content", "GOOGL": "Internet Content",
    "NFLX": "Entertainment", "AVGO": "Semiconductors", "QCOM": "Semiconductors", "INTC": "Semiconductors",
    "MU": "Semiconductors", "ARM": "Semiconductors", "TXN": "Semiconductors", "AMAT": "Semiconductor Equip",
    "LRCX": "Semiconductor Equip", "ADI": "Semiconductors", "IBM": "IT Services", "CSCO": "Communication Equip",
    "ORCL": "Software - Infra", "PLTR": "Software - App", "CRM": "Software - App", "ADBE": "Software - Infra",
    "SNOW": "Software - App", "NOW": "Software - App", "WDAY": "Software - App", "PANW": "Software - Infra",
    "CRWD": "Software - Infra", "DDOG": "Software - App", "NET": "Software - Infra", "COIN": "Software - Infra",
    "MSTR": "Software - App", "HOOD": "Software - Infra", "SQ": "Software - Infra", "PYPL": "Credit Services",
    "V": "Credit Services", "MA": "Credit Services", "AFRM": "Credit Services", "SOFI": "Credit Services",
    "DKNG": "Gambling", "UBER": "Software - App", "ABNB": "Travel Services", "ROKU": "Entertainment",
    "SHOP": "Software - App", "DIS": "Entertainment", "NKE": "Footwear & Accessories", "SBUX": "Restaurants",
    "MCD": "Restaurants", "WMT": "Discount Stores", "TGT": "Discount Stores", "COST": "Discount Stores",
    "HD": "Home Improvement", "LOW": "Home Improvement", "LULU": "Apparel Retail", "CMG": "Restaurants",
    "JPM": "Banks - Diversified", "BAC": "Banks - Diversified", "WFC": "Banks - Diversified", "GS": "Capital Markets",
    "MS": "Capital Markets", "C": "Banks - Diversified", "AXP": "Credit Services", "BLK": "Asset Management",
    "BA": "Aerospace & Defense", "CAT": "Farm & Heavy Const", "GE": "Specialty Ind Mach", "F": "Auto Manufacturers",
    "GM": "Auto Manufacturers", "XOM": "Oil & Gas Integrated", "CVX": "Oil & Gas Integrated", "COP": "Oil & Gas E&P",
    "OXY": "Oil & Gas E&P", "SLB": "Oil & Gas Equip", "HAL": "Oil & Gas Equip", "LLY": "Drug Manufacturers",
    "UNH": "Healthcare Plans", "JNJ": "Drug Manufacturers", "PFE": "Drug Manufacturers", "MRK": "Drug Manufacturers",
    "ABBV": "Drug Manufacturers", "BMY": "Drug Manufacturers", "AMGN": "Drug Manufacturers", "GILD": "Drug Manufacturers",
    "MRNA": "Biotechnology"
}

LIQUID_TICKERS = list(SECTOR_MAP.keys())

ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "SMH", "ARKK", "KRE", "XBI", "GDX",
    "EEM", "FXI", "EWZ", "HYG", "LQD", "UVXY", "BITO", "USO", "UNG", "TQQQ", "SQQQ", "SOXL", "SOXS"
]

# --- INITIALIZE SERVICES ---
drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
except Exception as e:
    finnhub_client = None

if "trades" not in st.session_state:
    st.session_state.trades = load_from_drive(drive_service) or [] if drive_service else []

if "scan_results" not in st.session_state:
    st.session_state.scan_results = [] 

if "scan_log" not in st.session_state:
    st.session_state.scan_log = []

if "current_ticker_index" not in st.session_state:
    st.session_state.current_ticker_index = 0
if "batch_complete" not in st.session_state:
    st.session_state.batch_complete = False

# Global Session
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

# --- TECHNICAL ANALYSIS HELPER ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SMART RETRY LOGIC ---
def safe_yfinance_call(func, *args, **kwargs):
    max_retries = 3
    base_wait = 2 
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "too many requests" in err_msg:
                wait_time = base_wait * (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1: return None
                time.sleep(1)
    return None

# --- EARNINGS SAFETY CHECK ---
def is_safe_from_earnings(ticker, expiration_date_str):
    if ticker in ETFS: return True, "ETF"
    if not finnhub_client: return True, "No API"
    try:
        time.sleep(0.5) 
        today = datetime.now().date()
        exp_date = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        earnings = finnhub_client.earnings_calendar(_from=today.strftime("%Y-%m-%d"), to=exp_date.strftime("%Y-%m-%d"), symbol=ticker)
        if earnings and 'earningsCalendar' in earnings and len(earnings['earningsCalendar']) > 0:
            return False, earnings['earningsCalendar'][0]['date']
        return True, "Clear" 
    except Exception as e:
        return True, "Error"

# --- MARKET HEALTH CHECK ---
def get_market_health():
    try:
        # Use Finnhub for price (safe)
        if finnhub_client:
            try:
                quote = finnhub_client.quote("SPY")
                current_price = quote['c']
            except:
                current_price = 0
        else:
            current_price = 0

        # Use Yahoo for SMA
        spy = yf.Ticker("SPY", session=session)
        hist = safe_yfinance_call(spy.history, period="1y")
        
        if hist is None or hist.empty: return True, current_price, 0
        
        if current_price == 0:
            current_price = hist['Close'].iloc[-1]
            
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        return current_price > sma_200, current_price, sma_200
    except: return True, 0, 0

# --- BULK DATA PROCESSING (FIXED) ---
def process_bulk_data(df, ticker):
    """
    Extracts single ticker data from the bulk DataFrame.
    """
    try:
        hist = pd.DataFrame()
        
        # KEY FIX: Handle the MultiIndex correctly based on group_by='ticker'
        if isinstance(df.columns, pd.MultiIndex):
            # If ticker is top level
            try:
                ticker_df = df[ticker].copy()
            except KeyError:
                return None
        else:
            # If flat (single ticker downloaded), columns are OHLC directly
            # This happens if batch size is 1 or only 1 succeeded
            ticker_df = df.copy()

        # Normalize Columns
        if 'Close' not in ticker_df.columns:
            return None
            
        hist['Close'] = ticker_df['Close']
        hist['Open'] = ticker_df['Open'] if 'Open' in ticker_df.columns else hist['Close']
        
        hist = hist.dropna()
        if hist.empty: return None

        current_price = hist['Close'].iloc[-1]
        
        if len(hist) > 1:
            prev_price = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_price) / prev_price) * 100
        else:
            change_pct = 0.0
        
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        current_hv = hist['HV'].iloc[-1] if not pd.isna(hist['HV'].iloc[-1]) else 0

        hist['SMA_100'] = hist['Close'].rolling(window=100).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])

        # BB
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['STD_20'] = hist['Close'].rolling(window=20).std()
        hist['BB_Upper'] = hist['SMA_20'] + (hist['STD_20'] * 2)
        hist['BB_Lower'] = hist['SMA_20'] - (hist['STD_20'] * 2)

        sma_200 = hist['SMA_200'].iloc[-1] if len(hist) >= 200 else current_price
        bb_upper = hist['BB_Upper'].iloc[-1] if not pd.isna(hist['BB_Upper'].iloc[-1]) else current_price * 1.05
        bb_lower = hist['BB_Lower'].iloc[-1] if not pd.isna(hist['BB_Lower'].iloc[-1]) else current_price * 0.95
        current_rsi = hist['RSI'].iloc[-1] if len(hist) >= 14 else 50

        is_uptrend = current_price > sma_200
        is_oversold_bb = current_price <= (bb_lower * 1.01)
        is_overbought_bb = current_price >= (bb_upper * 0.99)

        type_str = "(ETF)" if ticker in ETFS else "(Stock)"
        sector_str = SECTOR_MAP.get(ticker, "Unknown Sector")

        return {
            "price": current_price, "change_pct": change_pct, "hv": current_hv,
            "is_uptrend": is_uptrend, "rsi": current_rsi,
            "bb_lower": bb_lower, "bb_upper": bb_upper,
            "is_oversold_bb": is_oversold_bb, "is_overbought_bb": is_overbought_bb,
            "type_str": type_str, "sector_str": sector_str, "hist": hist
        }
    except Exception as e:
        return None

# --- TRADE LOGIC ---
def find_optimal_spread(ticker, stock_obj, current_price, current_hv, dev_mode=False):
    try:
        try:
            exps = stock_obj.options
        except: 
            time.sleep(1) # Tiny pause
            exps = stock_obj.options
        if not exps: return None, "No Options Chain"
        
        min_days = 14 if dev_mode else 25
        max_days = 60 if dev_mode else 50
        
        target_min_date = datetime.now() + timedelta(days=min_days)
        target_max_date = datetime.now() + timedelta(days=max_days)
        
        valid_exps = []
        for e in exps:
            try:
                edate = datetime.strptime(e, "%Y-%m-%d")
                if target_min_date <= edate <= target_max_date:
                    valid_exps.append(e)
            except: pass
        if not valid_exps: return None, "No Valid Expiry"
        best_exp = max(valid_exps) 
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days

        chain = safe_yfinance_call(stock_obj.option_chain, best_exp)
        if not chain: return None, "Failed to fetch Chain"
        
        puts = chain.puts
        atm_puts = puts[abs(puts['strike'] - current_price) == abs(puts['strike'] - current_price).min()]
        imp_vol = atm_puts.iloc[0]['impliedVolatility'] if not atm_puts.empty else (current_hv / 100.0)

        expected_move = current_price * imp_vol * np.sqrt(dte/365) * 0.75
        target_short_strike = current_price - expected_move
        
        otm_puts = puts[puts['strike'] <= target_short_strike].sort_values('strike', ascending=False)
        
        if otm_puts.empty:
            if dev_mode: otm_puts = puts[puts['strike'] < current_price].sort_values('strike', ascending=False)
            else:
                target_short_fallback = current_price * 0.95
                otm_puts = puts[puts['strike'] <= target_short_fallback].sort_values('strike', ascending=False)
        if otm_puts.empty: return None, "No Safe Strikes"

        width = 5.0
        if ticker in ETFS: min_credit = 0.50 
        else: min_credit = 0.70 
        if dev_mode: min_credit = 0.05

        best_spread = None
        rejection_reason = f"Credit < ${min_credit}"

        for index, short_row in otm_puts.iterrows():
            short_strike = short_row['strike']
            bid = short_row['bid']
            ask = short_row['ask']
            if ask <= 0: continue
            
            spread_width = ask - bid
            slippage_pct = spread_width / ask
            is_liquid = dev_mode or (spread_width <= 0.05) or (slippage_pct <= 0.20)
            if not is_liquid: 
                rejection_reason = "Illiquid Strikes"
                continue

            long_strike_target = short_strike - width
            long_leg = puts[abs(puts['strike'] - long_strike_target) < 0.2] 
            if long_leg.empty: continue 
            long_row = long_leg.iloc[0]
            mid_credit = bid - long_row['ask']
            
            if mid_credit >= min_credit:
                max_loss = width - mid_credit
                roi = (mid_credit / max_loss) * 100 if max_loss > 0 else 0
                iv = short_row['impliedVolatility'] * 100
                
                exp_date_obj = datetime.strptime(best_exp, "%Y-%m-%d")
                exp_date_str = exp_date_obj.strftime("%b %d, %Y")

                best_spread = {
                    "expiration_raw": best_exp, "expiration": exp_date_str, "dte": dte, 
                    "short": short_strike, "long": long_row['strike'],
                    "credit": mid_credit, "max_loss": max_loss, 
                    "iv": iv, "roi": roi, "em": expected_move
                }
                break 
        
        if best_spread and not dev_mode:
            is_safe, msg = is_safe_from_earnings(ticker, best_exp)
            if not is_safe: return None, f"Earnings on {msg}"
        
        if best_spread: return best_spread, None
        else: return None, rejection_reason

    except Exception as e: return None, f"Error: {str(e)}"

# --- PLOTTING ---
def plot_clean_sparkline(hist, short_strike, long_strike):
    fig, ax = plt.subplots(figsize=(4, 1.3)) 
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    last_60 = hist.tail(60).copy()
    dates = last_60.index
    if 'Open' not in last_60.columns: last_60['Open'] = last_60['Close']
    bar_colors = np.where(last_60['Close'] >= last_60['Open'], SUCCESS_COLOR, WARNING_COLOR)
    heights = last_60['Close'] - last_60['Open']
    bottoms = last_60['Open']
    ax.bar(dates, heights, bottom=bottoms, color=bar_colors, width=0.8, align='center', alpha=0.9)
    ax.axhline(y=short_strike, color=STRIKE_COLOR, linestyle='-', linewidth=1, alpha=0.9)
    ax.axhline(y=long_strike, color=STRIKE_COLOR, linestyle='-', linewidth=0.8, alpha=0.6)
    ax.fill_between(dates, long_strike, short_strike, color=STRIKE_COLOR, alpha=0.1)
    ax.grid(True, which='major', linestyle=':', color=GRID_COLOR, alpha=0.3)
    ax.axis('off') 
    plt.tight_layout(pad=0.1)
    return fig

# --- MAIN UI ---
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    try: st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except: st.write("**DR CAPITAL**")
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Spread Finder</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Scanner Settings")
    dev_mode = st.checkbox("Dev Mode (Bypass Filters)", value=False, help="Check this to test in bear markets.")
    if dev_mode: st.warning("DEV MODE ACTIVE: Safety Filters Disabled.")
    
    if st.button("Reset Scanner"):
        st.session_state.current_ticker_index = 0
        st.session_state.scan_results = []
        st.session_state.scan_log = []
        st.session_state.batch_complete = False
        st.rerun()

# --- BATCH SCAN LOGIC ---
total_tickers = len(LIQUID_TICKERS)
start_index = st.session_state.current_ticker_index
batch_size = 10
end_index = min(start_index + batch_size, total_tickers)

st.caption(f"Scanner Progress: {start_index}/{total_tickers} tickers scanned")
st.progress(start_index / total_tickers)

# --- SCAN BUTTON ---
btn_label = "Scan Next 10 Tickers" if start_index < total_tickers else "Scan Complete (Reset to Restart)"
if st.button(btn_label, disabled=(start_index >= total_tickers)):
    st.session_state.batch_complete = False 
    
    if start_index == 0:
        status = st.empty()
        status.info("Initializing Scanner...")
        time.sleep(2) 
        market_healthy, spy_price, spy_sma = get_market_health()
        status.empty()
        
        if not market_healthy and not dev_mode:
            st.error(f"BEAR REGIME DETECTED: SPY ${spy_price:.2f} < SMA ${spy_sma:.2f}")
            st.stop()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.markdown("---") 
    log_placeholder = st.empty()
    
    # --- 1. BULK DOWNLOAD ---
    current_batch_tickers = LIQUID_TICKERS[start_index:end_index]
    status_text.text(f"Fetching Bulk Data for {len(current_batch_tickers)} tickers...")
    
    try:
        # group_by='ticker' ensures data is structured as df[Ticker][Close]
        bulk_data = yf.download(current_batch_tickers, period="1y", group_by='ticker', progress=False)
        st.session_state.scan_log.append(f"[BATCH] Bulk download successful for {len(current_batch_tickers)} tickers")
    except Exception as e:
        bulk_data = None
        st.session_state.scan_log.append(f"[BATCH] Bulk download failed: {e}")

    # --- 2. LOOP ---
    for i, ticker in enumerate(current_batch_tickers):
        status_text.text(f"Analyzing {ticker}...")
        progress_bar.progress((i + 1) / len(current_batch_tickers))
        
        # Use data from Bulk Dataframe
        if bulk_data is not None and not bulk_data.empty:
            data = process_bulk_data(bulk_data, ticker)
        else:
            data = None 
            
        if not data: 
            st.session_state.scan_log.append(f"[{ticker}] Data Unavailable (Check connection)")
            continue
        
        time.sleep(random.uniform(1.0, 2.0))
        spread, reject_reason = find_optimal_spread(ticker, yf.Ticker(ticker, session=session), data['price'], data['hv'], dev_mode=dev_mode)
        
        if spread:
             st.session_state.scan_log.append(f"[{ticker}] FOUND TRADE | Credit: ${spread['credit']:.2f}")
             
             score = 0
             if spread['iv'] > (data['hv'] + 5.0): score += 30
             elif spread['iv'] > data['hv']: score += 15
             width = spread['short'] - spread['long']
             if width > 0:
                 credit_ratio = spread['credit'] / width
                 if credit_ratio >= 0.30: score += 20
                 elif credit_ratio >= 0.25: score += 15
                 elif credit_ratio >= 0.15: score += 10
             if data['is_uptrend']: score += 10
             
             if data['is_uptrend'] and data['is_oversold_bb']: score += 30
             elif data['is_uptrend'] and data['rsi'] < 45: score += 20
             elif data['rsi'] < 50: score += 10
             if data['is_overbought_bb'] or data['rsi'] > 70: score -= 20
             
             score += 20 
             display_score = min(score, 100.0)
             
             if dev_mode or display_score >= 50:
                 st.session_state.scan_results.append({
                     "ticker": ticker, "data": data, "spread": spread, 
                     "score": score, "display_score": display_score
                 })
        else:
            st.session_state.scan_log.append(f"[{ticker}] Skipped: {reject_reason}")
        
        log_text = "\n".join(st.session_state.scan_log[-10:])
        log_placeholder.code(log_text, language="text")

    st.session_state.current_ticker_index = end_index
    st.session_state.batch_complete = True 
    progress_bar.empty()
    status_text.empty()
    st.rerun() 

# --- DISPLAY LOGIC ---
if st.session_state.batch_complete and st.session_state.scan_results:
    sorted_results = sorted(st.session_state.scan_results, key=lambda x: x['score'], reverse=True)
    st.success(f"Total Opportunities Found: {len(sorted_results)}")
    cols = st.columns(3)
    
    for i, res in enumerate(sorted_results):
        t = res['ticker']
        d = res['data']
        s = res['spread']
        
        with cols[i % 3]:
            with st.container(border=True):
                pill_class = "price-pill-red" if d['change_pct'] < 0 else "price-pill-green"
                badge_text = "ELITE EDGE" if res['display_score'] >= 80 else "SOLID SETUP"
                badge_style = "border: 1px solid #00C853; color: #00C853;" if res['display_score'] >= 80 else "border: 1px solid #d4ac0d; color: #d4ac0d;"
                if dev_mode and res['display_score'] < 50:
                    badge_text = "TEST RESULT"
                    badge_style = "border: 1px solid gray; color: gray;"

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 22px; font-weight: 900; color: white; line-height: 1;">
                            {t} <span style="font-size: 12px; font-weight: 400; color: #aaa;">{d['type_str']}</span>
                        </div>
                        <div style="font-size: 11px; color: #888; margin-bottom: 4px;">{d['sector_str']}</div>
                        <div style="margin-top: 2px;"><span class="{pill_class}">${d['price']:.2f} ({d['change_pct']:.2f}%)</span></div>
                    </div>
                    <div class="strategy-badge" style="{badge_style}">{badge_text}</div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
                
                exp_dt = datetime.strptime(s['expiration_raw'], "%Y-%m-%d")
                exit_dt = exp_dt - timedelta(days=21)
                exit_str = exit_dt.strftime("%b %d, %Y")
                
                if d['is_uptrend'] and (d['is_oversold_bb'] or d['rsi'] < 45):
                    signal_html = "<span style='color: #00FFAA; font-weight: bold; font-size: 14px;'>BUY NOW (DIP)</span>"
                elif d['is_uptrend']:
                    signal_html = "<span style='color: #FFA726; font-weight: bold; font-size: 14px;'>WAIT (NEUTRAL)</span>"
                else:
                    signal_html = "<span style='color: #FF5252; font-weight: bold; font-size: 14px;'>PASS (TREND)</span>"

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
                            <div class="metric-value" style="color:#FF5252">${s['max_loss']:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-label">Expiry</div>
                    <div class="metric-value">{s['dte']} Days</div>
                    <div style="height: 8px;"></div>
                    <div class="metric-label">Action Signal</div>
                    <div>{signal_html}</div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                vc1, vc2 = st.columns([2, 1])
                with vc1: st.pyplot(plot_clean_sparkline(d['hist'], s['short'], s['long']), use_container_width=True)
                with vc2:
                    st.markdown(f"""
                    <div class="metric-label" style="text-align: right;">Edge Score</div>
                    <div class="metric-value" style="text-align: right; color: #d4ac0d;">{res['display_score']:.0f}</div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""<div class="roc-box"><span style="font-size:11px; color: #00c864; text-transform: uppercase;">Return on Capital</span><br><span style="font-size:18px; font-weight:800; color: #00c864;">{s['roi']:.2f}%</span></div>""", unsafe_allow_html=True)
                
                add_key = f"add_mode_{t}_{i}"
                st.write("") 
                if st.button(f"Add {t}", key=f"btn_{t}_{i}", use_container_width=True):
                    st.session_state[add_key] = True

                if st.session_state.get(add_key, False):
                    st.markdown("##### Size")
                    num = st.number_input(f"Contracts", min_value=1, value=1, key=f"c_{t}_{i}")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("OK", key=f"ok_{t}_{i}"):
                            new_trade = {
                                "id": f"{t}-{s['short']}-{s['expiration_raw']}",
                                "ticker": t, "contracts": num, 
                                "short_strike": s['short'], "long_strike": s['long'],
                                "expiration": s['expiration_raw'], "credit": s['credit'],
                                "entry_date": datetime.now().date().isoformat(),
                                "pnl_history": []
                            }
                            st.session_state.trades.append(new_trade)
                            if drive_service: save_to_drive(drive_service, st.session_state.trades)
                            st.toast(f"Added {t}")
                            del st.session_state[add_key]
                            st.rerun()
                    with cc2:
                        if st.button("X", key=f"no_{t}_{i}"):
                            del st.session_state[add_key]
                            st.rerun()

# --- ALWAYS SHOW LOG AT BOTTOM ---
if st.session_state.scan_log:
    with st.expander("Show Full Scanner Log", expanded=True):
        st.code("\n".join(st.session_state.scan_log), language="text")
