import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from scipy import stats
from duckduckgo_search import DDGS
import finnhub
import os
import re
from datetime import datetime, timedelta

# ---------------- CONFIGURATION ----------------
st.set_page_config(layout="wide", page_title="Macro Intelligence", page_icon="⚡")

# ---------------- CONSTANTS ----------------
SUCCESS_COLOR = "#00C853"  # Green
WARNING_COLOR = "#d32f2f"  # Red
NEUTRAL_COLOR = "#FFA726"  # Orange
BG_COLOR = "#0E1117"
CARD_COLOR = "#262730"
FINNHUB_KEY = "d5mgc39r01ql1f2p69c0d5mgc39r01ql1f2p69cg" 

# Name Mapping
TICKER_MAP = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLY': 'Discretionary', 'XLP': 'Staples',
    'XLI': 'Industrials', 'XLU': 'Utilities', 'XLC': 'Comm.',
    'XLB': 'Materials', 'XLRE': 'Real Estate',
    'SMH': 'Semis (Growth)', 'IBB': 'Biotech (Risk)', 'XHB': 'Homebuilders (Rates)', 'KRE': 'Regional Banks',
    'GLD': 'Gold', 'SLV': 'Silver', 'USO': 'Oil', 'TLT': '20Y Bonds',
    '^TNX': '10-Year Yield'
}

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; }}
    .metric-card {{
        background-color: {CARD_COLOR};
        border: 1px solid #333;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .metric-title {{ color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 5px; }}
    .metric-value {{ color: #FFF; font-size: 28px; font-weight: 800; margin: 0; line-height: 1.2; }}
    .sub-metric {{ font-size: 12px; color: #aaa; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
    .interpretation {{ font-size: 13px; color: #BBB; margin-top: 15px; padding-top: 10px; border-top: 1px solid #444; line-height: 1.4; }}
    .news-item {{ padding: 8px 0; border-bottom: 1px solid #333; font-size: 13px; }}
    .news-link {{ color: #58a6ff; text-decoration: none; font-weight: 500; }}
    .news-link:hover {{ text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ENGINES ----------------

@st.cache_data(ttl=300)
def fetch_market_data():
    tickers = {
        "Main": ["SPY", "QQQ", "IWM"],
        "Vol": ["^VIX", "^VVIX"],
        "Rates": ["^TNX", "TLT"], 
        "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "SMH", "IBB", "XHB", "KRE"],
        "Commodities": ["GLD", "SLV", "USO"]
    }
    all_ticks = [t for cat in tickers.values() for t in cat]
    try:
        data = yf.download(all_ticks, period="1y", interval="1d", progress=False)['Close']
        return data
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# --- BACKUP SEARCH ENGINE (Rescues broken API calls) ---
def fetch_macro_backup():
    """Uses DuckDuckGo to find CPI/Fed info if API fails"""
    backup = {'cpi': None, 'fed': None}
    
    try:
        # 1. Search for CPI
        cpi_results = DDGS().text("current US CPI inflation rate year over year actual", max_results=2)
        for r in cpi_results:
            # Look for percentage pattern like 3.1% or 2.9%
            match = re.search(r"(\d+\.\d+)%", r['body'])
            if match:
                val = float(match.group(1))
                # Basic sanity check (CPI won't be 50% or 0.1%)
                if 1.0 < val < 10.0:
                    backup['cpi'] = {
                        'val': f"{val}%",
                        'status': "Search Est.", # Indicate source
                        'date': "Latest Release"
                    }
                    break
        
        # 2. Search for Fed Date
        fed_results = DDGS().text("next FOMC meeting date 2025", max_results=2)
        for r in fed_results:
            # Very basic check if we find a month name
            body_lower = r['body'].lower()
            months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            for m in months:
                if m in body_lower and "meeting" in body_lower:
                    backup['fed'] = {'date': "Check Calendar", 'expectation': "Search Active"} 
                    break
                    
    except: pass
    return backup

@st.cache_data(ttl=3600)
def fetch_economic_events():
    """Tries API -> Falls back to Search -> Returns Safe Defaults"""
    
    # DEFAULT STATE (Clean Gray)
    result = {
        'cpi': {'val': '--', 'status': 'Unavailable', 'date': '-'},
        'fed': {'date': 'TBD', 'expectation': 'Unknown'}
    }

    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
        today = datetime.now()
        # Widen window to 4 months to catch quarterly delays
        start_date = (today - timedelta(days=120)).strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=90)).strftime('%Y-%m-%d')
        
        calendar = finnhub_client.economic_calendar(_from=start_date, to=end_date)
        
        # Guard against API key error returning dict with 'error' key instead of 'economicCalendar'
        if not calendar or 'economicCalendar' not in calendar:
            raise Exception("Invalid API Response")

        events = calendar.get('economicCalendar', [])
        
        # --- 1. CPI LOGIC ---
        # Robust filtering: Check for "CPI" or "Consumer Price Index"
        cpi_events = [
            e for e in events 
            if ('consumer price index' in e.get('event', '').lower() or 'cpi' in e.get('event', '').lower())
            and 'yoy' in e.get('event', '').lower()
            and e.get('country') == 'US'
        ]
        
        completed_cpi = [e for e in cpi_events if e.get('actual') is not None]
        
        if completed_cpi:
            latest = sorted(completed_cpi, key=lambda x: x['date'], reverse=True)[0]
            actual = latest.get('actual')
            estimate = latest.get('estimate')
            
            status = "In Line"
            if estimate is not None and actual is not None:
                if actual > estimate: status = "Hot (Bearish)"
                elif actual < estimate: status = "Cool (Bullish)"
            
            result['cpi'] = {
                'val': f"{actual}%",
                'status': status,
                'date': latest['date'].split(' ')[0]
            }
        
        # --- 2. FED LOGIC ---
        fed_events = [e for e in events if 'fed interest rate decision' in e.get('event', '').lower() and e['country'] == 'US']
        future_fed = [e for e in fed_events if e['date'] >= today.strftime('%Y-%m-%d')]
        
        if future_fed:
            next_meeting = sorted(future_fed, key=lambda x: x['date'])[0]
            
            est = next_meeting.get('estimate')
            prev = next_meeting.get('prev')
            
            expectation = "Hold Expected"
            if est is not None and prev is not None:
                if est < prev: expectation = "CUT Expected"
                elif est > prev: expectation = "HIKE Expected"
            
            result['fed'] = {
                'date': next_meeting['date'].split(' ')[0],
                'expectation': expectation
            }
            
    except Exception as e:
        # API Failed: Trigger Backup Search
        backup = fetch_macro_backup()
        if backup['cpi']: result['cpi'] = backup['cpi']
        if backup['fed']: result['fed'] = backup['fed']
        
    return result

@st.cache_data(ttl=1800)
def fetch_news_headlines():
    try:
        results = DDGS().news(keywords="market news finance", region="wt-wt", safesearch="off", max_results=5)
        return results
    except:
        return []

def calculate_metrics(data, eco_data):
    metrics = {}
    
    # 1. Volatility
    try:
        current_vix = data['^VIX'].iloc[-1]
        vix_history = data['^VIX'].dropna()
        iv_rank = stats.percentileofscore(vix_history, current_vix)
        metrics['vix'] = {'value': current_vix, 'rank': iv_rank}
    except:
        metrics['vix'] = {'value': 15.0, 'rank': 50}
    
    # 2. Trend
    try:
        spy = data['SPY']
        sma200 = spy.rolling(200).mean().iloc[-1]
        sma50 = spy.rolling(50).mean().iloc[-1]
        price = spy.iloc[-1]
        
        trend_state = "Neutral"
        if price > sma200:
            trend_state = "Bullish" if price > sma50 else "Bullish (Weak)"
        else:
            trend_state = "Bearish" if price < sma50 else "Bearish (Rally)"
            
        metrics['spy'] = {'price': price, 'sma200': sma200, 'sma50': sma50, 'state': trend_state}
    except:
        metrics['spy'] = {'price': 0, 'sma200': 0, 'sma50': 0, 'state': "Error"}
    
    # 3. Risk (Enhanced)
    try:
        offense = data['SMH'].pct_change(20).iloc[-1]
        defense = data['XLU'].pct_change(20).iloc[-1]
        metrics['risk_mode'] = "Risk On" if offense > defense else "Risk Off"
    except:
        metrics['risk_mode'] = "Neutral"

    # 4. Macro
    try:
        tnx = data['^TNX'].iloc[-1] 
        tnx_prev = data['^TNX'].iloc[-2]
        metrics['macro'] = {
            'tnx': tnx,
            'tnx_chg': tnx - tnx_prev,
            'cpi': eco_data['cpi'],
            'fed': eco_data['fed']
        }
    except:
        metrics['macro'] = {
            'tnx': 0.0, 'tnx_chg': 0.0, 
            'cpi': eco_data['cpi'], 
            'fed': eco_data['fed']
        }
    
    return metrics

# ---------------- VISUALIZATION ----------------

def plot_spy_chart_mpl(data):
    if 'SPY' not in data: return None
    
    df = data['SPY'].reset_index()
    df.columns = ['Date', 'Close']
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df = df.tail(126)
    
    fig, ax = plt.subplots(figsize=(8, 3.5)) 
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)
    ax.plot(df['Date'], df['SMA200'], color=SUCCESS_COLOR, linestyle='--', linewidth=1, label='200 SMA', alpha=0.9)
    ax.plot(df['Date'], df['SMA50'], color=NEUTRAL_COLOR, linestyle=':', linewidth=1, label='50 SMA', alpha=0.9)
    ax.plot(df['Date'], df['Close'], color='#FFF', linewidth=1.5, label='Price')
    ax.fill_between(df['Date'], df['Close'], df['Close'].min(), color='#FFF', alpha=0.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#888')
    ax.spines['left'].set_color('#888')
    ax.tick_params(axis='x', colors='#888', labelsize=8)
    ax.tick_params(axis='y', colors='#888', labelsize=8)
    ax.grid(True, which='major', linestyle='-', color='#444', alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='upper left', fontsize=8, frameon=False, labelcolor='#ccc')
    plt.tight_layout()
    return fig

def draw_vix_gauge(val, rank):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)
    color = SUCCESS_COLOR if 13 <= val <= 20 else (WARNING_COLOR if val < 13 or val > 30 else NEUTRAL_COLOR)
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 0, 180, width=0.10, color='#333'))
    max_val = 40
    angle = min(val, max_val) / max_val * 180
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 180 - angle, 180, width=0.10, color=color))
    ax.text(0.5, 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=28, fontweight='bold', color='white')
    ax.text(0.5, -0.1, f"IV Rank: {rank:.0f}%", ha='center', va='top', fontsize=10, color='#aaa')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ---------------- MAIN APP LAYOUT ----------------

# HEADER
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])
with header_col1:
    if os.path.exists("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG"):
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    else:
        st.markdown("<h2 style='color:white; margin:0;'>DR CAPITAL</h2>", unsafe_allow_html=True)
with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Macro Intelligence Hub</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Volatility & Trend Analysis</p>
    </div>
    """, unsafe_allow_html=True)
with header_col3:
    st.write("") 

st.markdown(f"<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

with st.spinner("Analyzing Global Markets..."):
    raw_data = fetch_market_data()
    eco_data = fetch_economic_events()
    if raw_data.empty:
        st.error("Market Data Feed Offline. Please refresh.")
        st.stop()
    metrics = calculate_metrics(raw_data, eco_data)
    news = fetch_news_headlines()

# ---------------- ROW 1: METRICS ----------------
c1, c2, c3 = st.columns(3)

# 1. VOLATILITY
with c1:
    vix_val = metrics['vix']['value']
    vix_rank = metrics['vix']['rank']
    if vix_val < 13:
        vix_tag, vix_col, vix_msg = "COMPLACENT", WARNING_COLOR, "Premiums are cheap. Risk/Reward poor."
    elif 13 <= vix_val <= 22:
        vix_tag, vix_col, vix_msg = "OPTIMAL", SUCCESS_COLOR, "Goldilocks zone. Sell premium."
    else:
        vix_tag, vix_col, vix_msg = "ELEVATED", NEUTRAL_COLOR, "High fear. Reduce size."
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Volatility Command</div>
        <div style="display:flex; justify-content:center; margin-top:10px;">
    """, unsafe_allow_html=True)
    st.pyplot(draw_vix_gauge(vix_val, vix_rank), use_container_width=True)
    st.markdown(f"""
        </div>
        <div style="text-align:center; font-weight:bold; color:{vix_col}; font-size:18px;">{vix_tag}</div>
        <div class="interpretation">
            <strong>Action:</strong> {vix_msg}<br>
            <span style="color:#666; font-size:11px;">Implied Vol Rank: {vix_rank:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. TREND
with c2:
    trend_state = metrics['spy']['state']
    t_col = SUCCESS_COLOR if "Bullish" in trend_state else WARNING_COLOR
    st.markdown(f"""
    <div class="metric-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="metric-title">Market Trend (SPY)</div>
            <div style="color:{t_col}; font-weight:bold;">{trend_state}</div>
        </div>
    """, unsafe_allow_html=True)
    st.pyplot(plot_spy_chart_mpl(raw_data), use_container_width=True)
    st.markdown(f"""
        <div class="interpretation" style="margin-top:0;">
            <span style="color:#666; font-size:11px;">Price vs 200 SMA determines Regime.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 3. MACRO BACKDROP
with c3:
    m = metrics['macro']
    tnx_color = WARNING_COLOR if m['tnx'] > 4.5 else "#eee"
    
    # Clean up display values
    cpi_val = m['cpi']['val']
    cpi_status = m['cpi']['status']
    cpi_color = SUCCESS_COLOR if "Cool" in cpi_status or "Line" in cpi_status else "#eee"
    if "Hot" in cpi_status: cpi_color = WARNING_COLOR
    
    fed_date = m['fed']['date']
    fed_status = m['fed']['expectation']
    fed_color = SUCCESS_COLOR if "CUT" in fed_status else "#eee"
    
    # HTML Layout - Flexbox for stability
    macro_html = f"""<div class="metric-card">
<div class="metric-title">Macro Backdrop</div>
<div style="display: flex; justify-content: space-between; margin-top: 20px; gap: 10px;">
<div style="flex: 1; border-right: 1px solid #444; padding-right: 10px;">
<div class="sub-metric">10-Year Yield</div>
<div class="metric-value" style="color:{tnx_color}">{m['tnx']:.2f}%</div>
<div style="font-size:12px; font-weight:bold; color:#888;">{m['tnx_chg']:+.2f} pts</div>
</div>
<div style="flex: 1; padding-left: 10px;">
<div class="sub-metric">Inflation (CPI)</div>
<div class="metric-value" style="color:#eee">{cpi_val}</div>
<div style="font-size:12px; font-weight:bold; color:{cpi_color}">{cpi_status}</div>
</div>
</div>
<hr style="border-top:1px solid #444; margin:15px 0;">
<div>
<div class="sub-metric">Next Fed Decision: <span style="color:#eee;">{fed_date}</span></div>
<div style="font-size:18px; font-weight:bold; color:{fed_color}; margin-top:5px;">{fed_status}</div>
</div>
</div>"""
    st.markdown(macro_html, unsafe_allow_html=True)

# ---------------- ROW 2: PLAYBOOK & NEWS ----------------
r2_col1, r2_col2 = st.columns([2, 1])
with r2_col1:
    st.subheader("🛡️ The Playbook")
    if vix_val < 13:
        strategy_title, strategy_body, strategy_color = "The Sniper Phase (Patience)", "Volatility is too low. Risk/Reward is poor. **Action:** Reduce trade frequency. Buy Debit Spreads instead.", WARNING_COLOR
    elif metrics['spy']['price'] < metrics['spy']['sma200']:
        strategy_title, strategy_body, strategy_color = "The Bunker Phase (Defense)", "Trend is broken. Selling puts is risky. **Action:** Sit in cash. Wait for Price > 200 SMA.", WARNING_COLOR
    elif 13 <= vix_val <= 25:
        strategy_title, strategy_body, strategy_color = "The Harvest Phase (Aggressive)", "Conditions are perfect. Trend is up, premiums are fair. **Action:** Sell 30-45 DTE Put Spreads.", SUCCESS_COLOR
    else:
        strategy_title, strategy_body, strategy_color = "The Storm Phase (Caution)", "Volatility is extreme. **Action:** Cut position size by 50%. Widen strikes.", NEUTRAL_COLOR

    st.markdown(f"""
    <div style="background-color: {strategy_color}15; border-left: 4px solid {strategy_color}; padding: 15px; border-radius: 4px;">
        <h3 style="margin:0; color: {strategy_color}; font-size:18px;">{strategy_title}</h3>
        <p style="margin:8px 0 0 0; color: #ccc; font-size:14px;">{strategy_body}</p>
    </div>
    """, unsafe_allow_html=True)

with r2_col2:
    st.subheader("🌍 Headlines")
    if news:
        for n in news[:3]: 
            st.markdown(f"""<div class="news-item"><a href="{n['url']}" target="_blank" class="news-link">{n['title']}</a></div>""", unsafe_allow_html=True)
    else:
        st.caption("No immediate headlines found.")

st.divider()

# ---------------- ROW 3: DETAILED TABLES ----------------
with st.expander("📊 View Raw Sector Performance Data (Rotation Analysis)"):
    try:
        sector_list = ['SMH', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'IBB', 'XHB', 'KRE']
        commodity_list = ['GLD', 'SLV', 'USO', 'TLT']
        full_list = sector_list + commodity_list
        available = [t for t in full_list if t in raw_data.columns]
        subset = raw_data[available]
        perf = subset.pct_change(20).iloc[-1] * 100
        perf_df = pd.DataFrame(perf).reset_index()
        perf_df.columns = ['Ticker', '20d Return %']
        perf_df['Name'] = perf_df['Ticker'].map(TICKER_MAP).fillna(perf_df['Ticker'])
        perf_df = perf_df[['Name', 'Ticker', '20d Return %']]
        perf_df = perf_df.sort_values('20d Return %', ascending=False)
        st.dataframe(perf_df.style.background_gradient(cmap='RdYlGn', subset=['20d Return %']).format("{:.2f}%", subset=['20d Return %']), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not calculate performance table: {e}")
