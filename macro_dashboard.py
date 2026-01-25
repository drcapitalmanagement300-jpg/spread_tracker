import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import altair as alt
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
def fetch_cpi_data():
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        calendar = finnhub_client.economic_calendar(_from=start_date, to=end_date)
        cpi_events = [e for e in calendar['economicCalendar'] if 'Consumer Price Index (YoY)' in e['event'] and e['country'] == 'US']
        
        if cpi_events:
            latest_cpi = sorted(cpi_events, key=lambda x: x['date'], reverse=True)[0]
            return {
                "actual": latest_cpi['actual'],
                "prev": latest_cpi['prev'],
                "date": latest_cpi['date']
            }
        return None
    except:
        return None

@st.cache_data(ttl=1800)
def fetch_news_headlines():
    try:
        results = DDGS().news(keywords="financial markets news", region="wt-wt", safesearch="off", max_results=5)
        return results
    except:
        return []

def calculate_metrics(data, cpi_data):
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
        tnx = data['^TNX'].iloc[-1] # 10 Year Yield
        tnx_prev = data['^TNX'].iloc[-2]
        
        cpi_val = cpi_data['actual'] if cpi_data else 3.0
        cpi_prev = cpi_data['prev'] if cpi_data else 3.0
        
        metrics['macro'] = {
            'tnx': tnx,
            'tnx_chg': tnx - tnx_prev,
            'cpi': cpi_val,
            'cpi_prev': cpi_prev,
            'cpi_date': cpi_data['date'] if cpi_data else "N/A"
        }
    except:
        metrics['macro'] = {'tnx': 4.0, 'tnx_chg': 0, 'cpi': 3.0, 'cpi_prev': 3.0, 'cpi_date': "N/A"}
    
    return metrics

# ---------------- VISUALIZATION ----------------

def plot_trend_altair(data):
    if 'SPY' not in data: return None
    df = data['SPY'].reset_index()
    df.columns = ['Date', 'Price']
    df['SMA200'] = df['Price'].rolling(200).mean()
    df['SMA50'] = df['Price'].rolling(50).mean()
    df = df.tail(126)
    
    base = alt.Chart(df).encode(x='Date:T')
    line = base.mark_line(color='#ffffff', strokeWidth=2).encode(
        y=alt.Y('Price', scale=alt.Scale(zero=False), title=None),
        tooltip=['Date', 'Price']
    )
    sma200 = base.mark_line(color=SUCCESS_COLOR, strokeDash=[5, 5]).encode(y='SMA200', tooltip=['SMA200'])
    sma50 = base.mark_line(color=NEUTRAL_COLOR, strokeDash=[2, 2]).encode(y='SMA50', tooltip=['SMA50'])
    
    chart = (line + sma200 + sma50).properties(
        height=200, 
        title="SPY Market Structure (Price vs 50/200 SMA)"
    ).configure_axis(
        grid=False, labelColor='#888', titleColor='#888'
    ).configure_view(strokeWidth=0)
    
    return chart

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

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    if os.path.exists("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG"):
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=120)
    else:
        st.markdown("<h2 style='color:white; margin:0;'>DR CAPITAL</h2>", unsafe_allow_html=True)

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin:0; padding:0; font-size: 28px;'>Macro Intelligence Hub</h1>
        <p style='margin:0; font-size: 14px; color: gray;'>Strategic Volatility & Trend Analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# LOAD DATA
with st.spinner("Analyzing Global Markets..."):
    raw_data = fetch_market_data()
    cpi_data = fetch_cpi_data()
    
    if raw_data.empty:
        st.error("Market Data Feed Offline. Please refresh.")
        st.stop()
        
    metrics = calculate_metrics(raw_data, cpi_data)
    news = fetch_news_headlines()

# ---------------- ROW 1: THE BIG THREE ----------------
c1, c2, c3 = st.columns(3)

# 1. VOLATILITY CARD
with c1:
    vix_val = metrics['vix']['value']
    vix_rank = metrics['vix']['rank']
    
    if vix_val < 13:
        vix_msg = "Premiums are cheap. Edge is low."
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
        t_msg = "Market is in an uptrend. Put Credit Spreads favored."
    else:
        t_col = WARNING_COLOR
        t_msg = "Market is trending down. Defensive cash favored."

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Market Trend (SPY)</div>
        <div class="metric-value" style="color:{t_col}">{trend_state}</div>
        <div style="font-size:14px; color:#ccc;">${spy_price:.2f}</div>
        <div class="interpretation">
            <strong>What it means:</strong> {t_msg}<br>
            <span style="color:#666; font-size:11px;">Price vs 200 SMA: {"Above" if spy_price > sma200 else "Below"}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    chart = plot_trend_altair(raw_data)
    if chart: st.altair_chart(chart, use_container_width=True)

# 3. MACRO BACKDROP (FIXED: NO INDENTATION)
with c3:
    m = metrics['macro']
    tnx_color = WARNING_COLOR if m['tnx'] > 4.5 else NEUTRAL_COLOR
    cpi_color = WARNING_COLOR if m['cpi'] > 3.5 else SUCCESS_COLOR
    
    # IMPORTANT: The string below is NOT indented to prevent Markdown code-block errors
    macro_html = f"""<div class="metric-card">
<div class="metric-title">Macro Backdrop</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:15px;">
<div>
<div class="mini-stat-label">10-Year Yield</div>
<div class="metric-value" style="font-size:22px; color:{tnx_color}">{m['tnx']:.2f}%</div>
<div style="font-size:11px; color:#888;">{m['tnx_chg']:+.2f}% change</div>
</div>
<div>
<div class="mini-stat-label">Inflation (CPI)</div>
<div class="metric-value" style="font-size:22px; color:{cpi_color}">{m['cpi']:.1f}%</div>
<div style="font-size:11px; color:#888;">Prev: {m['cpi_prev']:.1f}%</div>
</div>
</div>
<div class="interpretation">
<strong>Fed Watch:</strong> Higher yields hurt valuations. Higher CPI keeps the Fed hawkish.<br>
<span style="color:#666; font-size:11px;">Latest CPI Date: {m['cpi_date']}</span>
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
