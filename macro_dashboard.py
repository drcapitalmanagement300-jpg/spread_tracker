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
from datetime import datetime, timedelta

# ---------------- CONFIGURATION ----------------
st.set_page_config(layout="wide", page_title="Macro Intelligence", page_icon="⚡")

# ---------------- CONSTANTS ----------------
SUCCESS_COLOR = "#00C853"  # Green
WARNING_COLOR = "#d32f2f"  # Red
NEUTRAL_COLOR = "#FFA726"  # Orange
BG_COLOR = "#0E1117"
CARD_COLOR = "#262730"
TEXT_COLOR = "#FAFAFA"
FINNHUB_KEY = "d5mgc39r01ql1f2p69c0d5mgc39r01ql1f2p69cg" 

# Name Mapping
TICKER_MAP = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLY': 'Cons. Discretionary', 'XLP': 'Cons. Staples',
    'XLI': 'Industrials', 'XLU': 'Utilities', 'XLC': 'Communication',
    'XLB': 'Materials', 'XLRE': 'Real Estate',
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
    .metric-title {{ color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
    .metric-value {{ color: #FFF; font-size: 26px; font-weight: 800; margin: 5px 0; }}
    .metric-delta {{ font-size: 14px; font-weight: bold; }}
    .interpretation {{ font-size: 13px; color: #BBB; margin-top: 10px; padding-top: 10px; border-top: 1px solid #444; line-height: 1.4; }}
    .mini-stat-label {{ font-size: 12px; color: #888; }}
    .mini-stat-val {{ font-size: 16px; font-weight: bold; color: #eee; }}
    .news-item {{ padding: 10px; border-bottom: 1px solid #333; }}
    .news-title {{ font-weight: bold; color: #58a6ff; text-decoration: none; }}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ENGINES ----------------

@st.cache_data(ttl=300)
def fetch_market_data():
    """Fetches comprehensive market data"""
    tickers = {
        "Main": ["SPY", "QQQ", "IWM"],
        "Vol": ["^VIX", "^VVIX"],
        "Rates": ["^TNX", "TLT"], 
        "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE"],
        "Commodities": ["GLD", "SLV", "USO"]
    }
    all_ticks = [t for cat in tickers.values() for t in cat]
    
    try:
        data = yf.download(all_ticks, period="1y", interval="1d", progress=False)['Close']
        return data
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_economic_events():
    """Fetches CPI and Fed Rate Decisions"""
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
        
        # Look back 2 months for CPI, forward 2 months for Fed
        today = datetime.now()
        start_date = (today - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=60)).strftime('%Y-%m-%d')
        
        calendar = finnhub_client.economic_calendar(_from=start_date, to=end_date)
        events = calendar.get('economicCalendar', [])
        
        # 1. FIND CPI
        cpi_data = {'val': 'N/A', 'status': 'Unknown', 'date': 'N/A'}
        # Filter for US CPI YoY
        cpi_events = [e for e in events if 'Consumer Price Index (YoY)' in e['event'] and e['country'] == 'US']
        
        # Get the latest COMPLETED event (has an 'actual' value)
        completed_cpi = [e for e in cpi_events if e.get('actual') is not None]
        
        if completed_cpi:
            latest = sorted(completed_cpi, key=lambda x: x['date'], reverse=True)[0]
            actual = latest.get('actual')
            estimate = latest.get('estimate')
            
            status = "As Expected"
            if estimate:
                if actual > estimate: status = "Heating Up (Bearish)"
                elif actual < estimate: status = "Cooling (Bullish)"
            
            cpi_data = {
                'val': actual,
                'prev': latest.get('prev'),
                'status': status,
                'date': latest['date'].split(' ')[0] # Remove time
            }

        # 2. FIND FED DECISION
        fed_data = {'date': 'N/A', 'expectation': 'Wait & See'}
        # Look for Fed Rate Decision
        fed_events = [e for e in events if 'Fed Interest Rate Decision' in e['event'] and e['country'] == 'US']
        
        # Find next FUTURE event
        future_fed = [e for e in fed_events if e['date'] >= today.strftime('%Y-%m-%d')]
        
        if future_fed:
            next_meeting = sorted(future_fed, key=lambda x: x['date'])[0]
            
            # Simple logic: Compare 'estimate' (consensus) vs 'prev'
            est_rate = next_meeting.get('estimate')
            prev_rate = next_meeting.get('prev')
            
            expectation = "Hold"
            if est_rate and prev_rate:
                if est_rate < prev_rate: expectation = "CUT Expected (Bullish)"
                elif est_rate > prev_rate: expectation = "HIKE Expected (Bearish)"
                
            fed_data = {
                'date': next_meeting['date'].split(' ')[0],
                'expectation': expectation
            }
            
        return {'cpi': cpi_data, 'fed': fed_data}
        
    except:
        return {'cpi': {'val': 'N/A', 'status': 'Error', 'date': 'N/A'}, 'fed': {'date': 'N/A', 'expectation': 'Unknown'}}

@st.cache_data(ttl=1800)
def fetch_news_headlines():
    try:
        results = DDGS().news(keywords="financial markets news", region="wt-wt", safesearch="off", max_results=5)
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
            trend_state = "Bullish" if price > sma50 else "Bullish (Pullback)"
        else:
            trend_state = "Bearish" if price < sma50 else "Bearish (Correction)"
            
        metrics['spy'] = {'price': price, 'sma200': sma200, 'sma50': sma50, 'state': trend_state}
    except:
        metrics['spy'] = {'price': 0, 'sma200': 0, 'sma50': 0, 'state': "Error"}
    
    # 3. Risk
    try:
        offense = data['XLK'].pct_change(20).iloc[-1]
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
        # Fallback if Yahoo fails
        metrics['macro'] = {
            'tnx': 0.0, 'tnx_chg': 0.0, 
            'cpi': eco_data['cpi'], 
            'fed': eco_data['fed']
        }
    
    return metrics

# ---------------- VISUALIZATION ----------------

def plot_spy_chart_mpl(data):
    """Matplotlib Bar Chart for SPY - Consistent with Dashboard"""
    if 'SPY' not in data: return None
    
    # Prepare Data
    df = data['SPY'].reset_index()
    df.columns = ['Date', 'Close']
    # Create synthetic OHLC for visualization purposes (approximation since we only have Close)
    # Note: In a real scenario, fetching OHLC is better, but this keeps data load light
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    
    # Filter to last 6 months
    df = df.tail(126)
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)
    
    # Plot SMAs
    ax.plot(df['Date'], df['SMA200'], color=SUCCESS_COLOR, linestyle='--', linewidth=1, label='200 SMA', alpha=0.8)
    ax.plot(df['Date'], df['SMA50'], color=NEUTRAL_COLOR, linestyle=':', linewidth=1, label='50 SMA', alpha=0.8)
    
    # Plot Price (White Line for clarity on macro view, cleaner than bars for 6m timeframe)
    ax.plot(df['Date'], df['Close'], color='#FFF', linewidth=1.5, label='Price')
    
    # Fill area below price to simulate "mountain" chart
    ax.fill_between(df['Date'], df['Close'], df['Close'].min(), color='#FFF', alpha=0.05)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#888')
    ax.spines['left'].set_color('#888')
    ax.tick_params(axis='x', colors='#888', labelsize=8)
    ax.tick_params(axis='y', colors='#888', labelsize=8)
    ax.grid(True, which='major', linestyle=':', color='#444', alpha=0.3)
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

# 1. CONSISTENT HEADER
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

# LOAD DATA
with st.spinner("Analyzing Global Markets..."):
    raw_data = fetch_market_data()
    eco_data = fetch_economic_events()
    
    if raw_data.empty:
        st.error("Market Data Feed Offline. Please refresh.")
        st.stop()
        
    metrics = calculate_metrics(raw_data, eco_data)
    news = fetch_news_headlines()

# ---------------- ROW 1: THE BIG THREE ----------------
c1, c2, c3 = st.columns(3)

# 1. VOLATILITY CARD
with c1:
    vix_val = metrics['vix']['value']
    vix_rank = metrics['vix']['rank']
    
    if vix_val < 13:
        vix_msg = "Premiums are cheap. Low edge."
        vix_tag = "COMPLACENT"
        vix_col = WARNING_COLOR
    elif 13 <= vix_val <= 22:
        vix_msg = "Optimal zone. Premiums are fair."
        vix_tag = "OPTIMAL"
        vix_col = SUCCESS_COLOR
    elif vix_val > 22:
        vix_msg = "Fear is high. Size down."
        vix_tag = "ELEVATED"
        vix_col = NEUTRAL_COLOR
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Volatility Regime (VIX)</div>
        <div style="display:flex; justify-content:center;">
    """, unsafe_allow_html=True)
    st.pyplot(draw_vix_gauge(vix_val, vix_rank), use_container_width=True)
    st.markdown(f"""
        </div>
        <div style="text-align:center; font-weight:bold; color:{vix_col}; margin-bottom:5px;">{vix_tag}</div>
        <div class="interpretation">
            <strong>What it means:</strong> {vix_msg}<br>
            <span style="color:#666; font-size:11px;">IV Rank: {vix_rank:.0f}% (Higher = Expensive Options)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 2. TREND CARD
with c2:
    trend_state = metrics['spy']['state']
    spy_price = metrics['spy']['price']
    sma200 = metrics['spy']['sma200']
    
    if "Bullish" in trend_state:
        t_col = SUCCESS_COLOR
        t_msg = "Market is in an uptrend."
    else:
        t_col = WARNING_COLOR
        t_msg = "Market is trending down."

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Market Trend (SPY)</div>
        <div class="metric-value" style="color:{t_col}">{trend_state}</div>
        <div style="font-size:14px; color:#ccc;">${spy_price:.2f}</div>
        <div class="interpretation">
            <strong>Structure:</strong> {t_msg}<br>
            <span style="color:#666; font-size:11px;">Price vs 200 SMA: {"Above" if spy_price > sma200 else "Below"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # REPLACED ALTAIR WITH MATPLOTLIB
    st.pyplot(plot_spy_chart_mpl(raw_data), use_container_width=True)

# 3. MACRO BACKDROP
with c3:
    m = metrics['macro']
    tnx_color = WARNING_COLOR if m['tnx'] > 4.5 else NEUTRAL_COLOR
    
    # CPI Color logic
    cpi_status_color = NEUTRAL_COLOR
    if "Cooling" in m['cpi']['status']: cpi_status_color = SUCCESS_COLOR
    elif "Heating" in m['cpi']['status']: cpi_status_color = WARNING_COLOR
    
    fed_status_color = NEUTRAL_COLOR
    if "CUT" in m['fed']['expectation']: fed_status_color = SUCCESS_COLOR
    elif "HIKE" in m['fed']['expectation']: fed_status_color = WARNING_COLOR
    
    # NOTE: NO INDENTATION in HTML string to prevent markdown errors
    macro_html = f"""<div class="metric-card">
<div class="metric-title">Macro Backdrop</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:15px;">
<div>
<div class="mini-stat-label">10-Year Yield</div>
<div class="metric-value" style="font-size:22px; color:{tnx_color}">{m['tnx']:.2f}%</div>
<div style="font-size:11px; color:#888;">{m['tnx_chg']:+.2f}% change</div>
</div>
<div>
<div class="mini-stat-label">Inflation (CPI)</div>
<div class="metric-value" style="font-size:22px; color:#eee">{m['cpi']['val']}</div>
<div style="font-size:11px; color:{cpi_status_color}; font-weight:bold;">{m['cpi']['status']}</div>
</div>
</div>
<hr style="border-top:1px solid #444; margin:10px 0;">
<div>
<div class="mini-stat-label">Next Fed Decision: <span style="color:#eee; font-weight:bold;">{m['fed']['date']}</span></div>
<div style="font-size:14px; margin-top:2px;">Outlook: <span style="color:{fed_status_color}; font-weight:bold;">{m['fed']['expectation']}</span></div>
</div>
<div class="interpretation" style="border:0; padding-top:5px;">
<span style="color:#666; font-size:11px;">CPI Date: {m['cpi']['date']}</span>
</div>
</div>"""
    st.markdown(macro_html, unsafe_allow_html=True)

# ---------------- ROW 2: PLAYBOOK & NEWS ----------------
r2_col1, r2_col2 = st.columns([2, 1])

with r2_col1:
    st.subheader("🛡️ The Playbook")
    
    if vix_val < 13:
        strategy_title = "The Sniper Phase (Patience)"
        strategy_body = "Volatility is too low. Risk/Reward is poor. **Action:** Reduce trade frequency. Buy Debit Spreads instead."
        strategy_color = WARNING_COLOR
    elif metrics['spy']['price'] < metrics['spy']['sma200']:
        strategy_title = "The Bunker Phase (Defense)"
        strategy_body = "Trend is broken. Selling puts is risky. **Action:** Sit in cash. Wait for Price > 200 SMA."
        strategy_color = WARNING_COLOR
    elif 13 <= vix_val <= 25:
        strategy_title = "The Harvest Phase (Aggressive)"
        strategy_body = "Conditions are perfect. Trend is up, premiums are fair. **Action:** Sell 30-45 DTE Put Spreads."
        strategy_color = SUCCESS_COLOR
    else:
        strategy_title = "The Storm Phase (Caution)"
        strategy_body = "Volatility is extreme. **Action:** Cut position size by 50%. Widen strikes."
        strategy_color = NEUTRAL_COLOR

    st.markdown(f"""
    <div style="background-color: {strategy_color}20; border-left: 5px solid {strategy_color}; padding: 15px; border-radius: 4px; height: 100%;">
        <h3 style="margin:0; color: {strategy_color};">{strategy_title}</h3>
        <p style="margin:10px 0 0 0; color: #ddd;">{strategy_body}</p>
    </div>
    """, unsafe_allow_html=True)

with r2_col2:
    st.subheader("🌍 Headlines")
    if news:
        for n in news[:3]: 
            st.markdown(f"""
            <div style="font-size:12px; margin-bottom:8px; border-left: 2px solid #444; padding-left:8px;">
                <a href="{n['url']}" target="_blank" style="color:#58a6ff; text-decoration:none;">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No immediate headlines found.")

st.divider()

# ---------------- ROW 3: DETAILED TABLES ----------------
with st.expander("📊 View Raw Sector Performance Data"):
    st.caption("20-Day Performance (Identifying Rotation)")
    
    try:
        sector_list = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU', 'XLC', 'XLB', 'XLRE']
        commodity_list = ['GLD', 'SLV', 'USO', 'TLT', '^TNX']
        
        full_list = sector_list + commodity_list
        available = [t for t in full_list if t in raw_data.columns]
        
        subset = raw_data[available]
        perf = subset.pct_change(20).iloc[-1] * 100
        
        perf_df = pd.DataFrame(perf).reset_index()
        perf_df.columns = ['Ticker', '20d Return %']
        perf_df['Name'] = perf_df['Ticker'].map(TICKER_MAP).fillna(perf_df['Ticker'])
        perf_df = perf_df[['Name', 'Ticker', '20d Return %']]
        perf_df = perf_df.sort_values('20d Return %', ascending=False)
        
        st.dataframe(
            perf_df.style.background_gradient(cmap='RdYlGn', subset=['20d Return %'])
            .format("{:.2f}%", subset=['20d Return %']), 
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.error(f"Could not calculate performance table: {e}")
