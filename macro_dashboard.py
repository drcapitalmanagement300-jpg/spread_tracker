import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import altair as alt
from scipy import stats
from duckduckgo_search import DDGS

# ---------------- CONFIGURATION ----------------
st.set_page_config(layout="wide", page_title="Macro Intelligence", page_icon="⚡")

# ---------------- CONSTANTS ----------------
SUCCESS_COLOR = "#00C853"  # Green
WARNING_COLOR = "#d32f2f"  # Red
NEUTRAL_COLOR = "#FFA726"  # Orange
BG_COLOR = "#0E1117"
CARD_COLOR = "#262730"
TEXT_COLOR = "#FAFAFA"

# Custom CSS for "Cards"
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; }}
    .metric-card {{
        background-color: {CARD_COLOR};
        border: 1px solid #333;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }}
    .metric-title {{ color: #888; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
    .metric-value {{ color: #FFF; font-size: 26px; font-weight: 800; margin: 5px 0; }}
    .metric-delta {{ font-size: 14px; font-weight: bold; }}
    .interpretation {{ font-size: 13px; color: #BBB; margin-top: 10px; padding-top: 10px; border-top: 1px solid #444; line-height: 1.4; }}
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
        "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU"]
    }
    all_ticks = [t for cat in tickers.values() for t in cat]
    
    # Download 1 year of data for percentile calculations
    data = yf.download(all_ticks, period="1y", interval="1d", progress=False)['Close']
    return data

@st.cache_data(ttl=1800)
def fetch_news_headlines():
    """Uses DuckDuckGo to get top market news summaries"""
    try:
        results = DDGS().news(keywords="stock market news today", region="wt-wt", safesearch="off", max_results=5)
        return results
    except:
        return []

def calculate_metrics(data):
    """Computes derived metrics like IV Rank, SMA alignment, and Sector Rotation"""
    metrics = {}
    
    # 1. Volatility Context (IV Percentile)
    current_vix = data['^VIX'].iloc[-1]
    vix_history = data['^VIX'].dropna()
    # Scipy: Calculate percentile rank of current VIX relative to last year
    iv_rank = stats.percentileofscore(vix_history, current_vix)
    
    metrics['vix'] = {
        'value': current_vix,
        'rank': iv_rank,
        'change': current_vix - data['^VIX'].iloc[-2]
    }
    
    # 2. Trend Alignment (SPY)
    spy = data['SPY']
    sma200 = spy.rolling(200).mean().iloc[-1]
    sma50 = spy.rolling(50).mean().iloc[-1]
    price = spy.iloc[-1]
    
    trend_state = "Neutral"
    if price > sma200:
        trend_state = "Bullish" if price > sma50 else "Bullish (Pullback)"
    else:
        trend_state = "Bearish" if price < sma50 else "Bearish (Correction)"
        
    metrics['spy'] = {
        'price': price,
        'sma200': sma200,
        'sma50': sma50,
        'state': trend_state
    }
    
    # 3. Sector Risk (Risk On/Off)
    # Compare 20-day returns of Offense (Tech XLK) vs Defense (Utilities XLU)
    offense = data['XLK'].pct_change(20).iloc[-1]
    defense = data['XLU'].pct_change(20).iloc[-1]
    
    metrics['risk_mode'] = "Risk On" if offense > defense else "Risk Off"
    
    return metrics

# ---------------- VISUALIZATION ----------------

def plot_trend_altair(data):
    """Interactive Altair Chart for SPY Trends"""
    df = data['SPY'].reset_index()
    df.columns = ['Date', 'Price']
    df['SMA200'] = df['Price'].rolling(200).mean()
    df['SMA50'] = df['Price'].rolling(50).mean()
    
    # Filter to last 6 months for clarity
    df = df.tail(126)
    
    base = alt.Chart(df).encode(x='Date:T')
    
    line = base.mark_line(color='#ffffff', strokeWidth=2).encode(
        y=alt.Y('Price', scale=alt.Scale(zero=False), title=None),
        tooltip=['Date', 'Price']
    )
    
    sma200 = base.mark_line(color=SUCCESS_COLOR, strokeDash=[5, 5]).encode(y='SMA200', tooltip=['SMA200'])
    sma50 = base.mark_line(color=NEUTRAL_COLOR, strokeDash=[2, 2]).encode(y='SMA50', tooltip=['SMA50'])
    
    chart = (line + sma200 + sma50).properties(
        height=250, 
        title="SPY Market Structure (Price vs 50/200 SMA)"
    ).configure_axis(
        grid=False, labelColor='#888', titleColor='#888'
    ).configure_view(strokeWidth=0)
    
    return chart

def draw_vix_gauge(val, rank):
    """Matplotlib Gauge showing VIX Value AND IV Rank"""
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)
    
    # Logic: 0-40 scale
    color = SUCCESS_COLOR if 13 <= val <= 20 else (WARNING_COLOR if val < 13 or val > 30 else NEUTRAL_COLOR)
    
    # Background Arc
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 0, 180, width=0.10, color='#333'))
    
    # Value Arc
    max_val = 40
    angle = min(val, max_val) / max_val * 180
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 180 - angle, 180, width=0.10, color=color))
    
    # Text
    ax.text(0.5, 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=28, fontweight='bold', color='white')
    ax.text(0.5, -0.1, f"IV Rank: {rank:.0f}%", ha='center', va='top', fontsize=10, color='#aaa')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ---------------- MAIN APP LAYOUT ----------------

st.title("Macro Intelligence Hub")
st.markdown("Contextual analysis for volatility and credit spread positioning.")
st.markdown(f"<hr style='border-top: 1px solid #333;'>", unsafe_allow_html=True)

# LOAD DATA
with st.spinner("Analyzing Global Markets..."):
    try:
        raw_data = fetch_market_data()
        metrics = calculate_metrics(raw_data)
        news = fetch_news_headlines()
    except Exception as e:
        st.error(f"Data Feed Error: {e}")
        st.stop()

# ---------------- ROW 1: THE BIG THREE ----------------
c1, c2, c3 = st.columns(3)

# 1. VOLATILITY CARD
with c1:
    vix_val = metrics['vix']['value']
    vix_rank = metrics['vix']['rank']
    
    # Interpretation Logic
    if vix_val < 13:
        vix_msg = "Premiums are cheap. It is hard to find edge. Risk/Reward is poor."
        vix_tag = "COMPLACENT"
        vix_col = WARNING_COLOR
    elif 13 <= vix_val <= 22:
        vix_msg = "Optimal zone. Premiums are fair, and moves are predictable."
        vix_tag = "OPTIMAL"
        vix_col = SUCCESS_COLOR
    elif vix_val > 22:
        vix_msg = "Fear is high. Premiums are juicy but moves are violent. Size down."
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
        t_msg = "Market is in an uptrend. Put Credit Spreads have the wind at their back."
    else:
        t_col = WARNING_COLOR
        t_msg = "Market is trending down. Selling Puts is dangerous (catching knives)."

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
    
    # Render Altair Chart inside the column
    st.altair_chart(plot_trend_altair(raw_data), use_container_width=True)

# 3. RISK/SECTOR CARD
with c3:
    risk_mode = metrics['risk_mode']
    
    if risk_mode == "Risk On":
        r_col = SUCCESS_COLOR
        r_msg = "Money is flowing into Tech & Discretionary. Investors are confident."
    else:
        r_col = NEUTRAL_COLOR
        r_msg = "Money is rotating into Utilities & Staples. Investors are defensive."
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Internal Rotation</div>
        <div class="metric-value" style="color:{r_col}">{risk_mode}</div>
        <div class="interpretation">
            <strong>What it means:</strong> {r_msg}<br>
            <span style="color:#666; font-size:11px;">Comparing XLK (Offense) vs XLU (Defense)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # News Feed using DuckDuckGo
    st.markdown("<div style='margin-top:10px; font-size:14px; font-weight:bold; color:#ddd;'>🌍 Why is the market moving?</div>", unsafe_allow_html=True)
    
    if news:
        for n in news[:3]: # Show top 3 headlines
            st.markdown(f"""
            <div style="font-size:12px; margin-bottom:8px; border-left: 2px solid #444; padding-left:8px;">
                <a href="{n['url']}" target="_blank" style="color:#58a6ff; text-decoration:none;">{n['title']}</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No immediate headlines found.")

# ---------------- ROW 2: TRADER'S PLAYBOOK ----------------

st.subheader("🛡️ The Playbook: How to Trade This Market")

# Determine Strategy based on Matrix
if vix_val < 13:
    strategy_title = "The Sniper Phase (Patience)"
    strategy_body = "Volatility is too low. Option sellers are getting paid pennies to take steamroller risk. **Action:** Reduce trade frequency. Buy Debit Spreads instead, or wait for a pullback."
    strategy_color = WARNING_COLOR
elif metrics['spy']['price'] < metrics['spy']['sma200']:
    strategy_title = "The Bunker Phase (Defense)"
    strategy_body = "The long-term trend is broken. Selling puts here is extremely risky. **Action:** Sit in cash. Do not average down. Wait for Price > 200 SMA."
    strategy_color = WARNING_COLOR
elif 13 <= vix_val <= 25:
    strategy_title = "The Harvest Phase (Aggressive)"
    strategy_body = "Conditions are perfect. Trend is up, premiums are fair. **Action:** Sell 30-45 DTE Put Spreads on high-quality tickers (check Spread Finder)."
    strategy_color = SUCCESS_COLOR
else:
    strategy_title = "The Storm Phase (Caution)"
    strategy_body = "Volatility is extreme. Prices will swing wildly. **Action:** Cut position size by 50%. Widen your strikes to stay safe."
    strategy_color = NEUTRAL_COLOR

st.markdown(f"""
<div style="background-color: {strategy_color}20; border-left: 5px solid {strategy_color}; padding: 15px; border-radius: 4px;">
    <h3 style="margin:0; color: {strategy_color};">{strategy_title}</h3>
    <p style="margin:10px 0 0 0; color: #ddd;">{strategy_body}</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- ROW 3: DETAILED TABLES ----------------
with st.expander("📊 View Raw Sector Performance Data"):
    st.caption("20-Day Performance (Identifying Rotation)")
    
    try:
        available_sectors = [s for s in ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLU'] if s in raw_data.columns]
        sectors = raw_data[available_sectors]
        perf = sectors.pct_change(20).iloc[-1] * 100
        perf_df = pd.DataFrame(perf).reset_index()
        perf_df.columns = ['Sector', '20d Return %']
        perf_df = perf_df.sort_values('20d Return %', ascending=False)
        
        # PATCHED: Explicit formatting with subset
        st.dataframe(
            perf_df.style.background_gradient(cmap='RdYlGn', subset=['20d Return %'])
            .format("{:.2f}%", subset=['20d Return %']), 
            use_container_width=True,
            hide_index=True
        )
    except Exception as e:
        st.error(f"Could not calculate sector table: {e}")
