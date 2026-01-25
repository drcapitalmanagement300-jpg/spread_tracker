import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ---------------- CONFIGURATION ----------------
st.set_page_config(layout="wide", page_title="Macro Command Center", page_icon="⚡")

# ---------------- CONSTANTS & STYLE ----------------
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#FFA726"
BG_COLOR = "#0E1117"
CARD_COLOR = "#262730"
TEXT_COLOR = "#FAFAFA"

# Inject Custom CSS to match your main dashboard
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; }}
    .macro-card {{
        background-color: {CARD_COLOR};
        border-radius: 6px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 10px;
        text-align: center;
    }}
    .macro-label {{ font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
    .macro-value {{ font-size: 32px; font-weight: 900; margin: 10px 0; color: #FFF; }}
    .macro-status {{ font-size: 14px; font-weight: bold; padding: 4px 12px; border-radius: 12px; display: inline-block; }}
    .status-green {{ background-color: rgba(0, 200, 83, 0.2); color: {SUCCESS_COLOR}; }}
    .status-red {{ background-color: rgba(211, 47, 47, 0.2); color: {WARNING_COLOR}; }}
    .status-orange {{ background-color: rgba(255, 167, 38, 0.2); color: {NEUTRAL_COLOR}; }}
</style>
""", unsafe_allow_html=True)

# ---------------- DATA ENGINE ----------------
@st.cache_data(ttl=300)
def get_macro_data():
    tickers = {
        "Indices": ["SPY"],
        "Volatility": ["^VIX"],
        "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLC", "XLB", "XLRE"]
    }
    all_ticks = [t for cat in tickers.values() for t in cat]
    try:
        data = yf.download(all_ticks, period="6mo", interval="1d", progress=False)['Close']
        return data, tickers['Sectors']
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame(), []

def analyze_regime(data):
    if data.empty: return {}

    # 1. VIX LOGIC
    try: vix = data['^VIX'].iloc[-1]
    except: vix = 15.0

    if vix < 13:
        vix_txt, vix_col, vix_css = "COMPLACENT", WARNING_COLOR, "status-red"
        vix_msg = "Premiums are cheap. Low edge."
    elif 13 <= vix <= 20:
        vix_txt, vix_col, vix_css = "OPTIMAL", SUCCESS_COLOR, "status-green"
        vix_msg = "Goldilocks zone for selling."
    elif 20 < vix <= 30:
        vix_txt, vix_col, vix_css = "ELEVATED", NEUTRAL_COLOR, "status-orange"
        vix_msg = "High premiums. Move fast."
    else:
        vix_txt, vix_col, vix_css = "PANIC", WARNING_COLOR, "status-red"
        vix_msg = "Extreme fear. Reduce size."

    # 2. TREND LOGIC (SPY)
    try:
        spy = data['SPY']
        sma200 = spy.rolling(200).mean().iloc[-1]
        price = spy.iloc[-1]
        
        if price > sma200:
            trend_txt, trend_col, trend_css = "BULLISH", SUCCESS_COLOR, "status-green"
            trend_msg = "SPY > 200 SMA. Safe to sell puts."
        else:
            trend_txt, trend_col, trend_css = "BEARISH", WARNING_COLOR, "status-red"
            trend_msg = "SPY < 200 SMA. Cash is king."
    except:
        price, sma200 = 0, 0
        trend_txt, trend_col, trend_css, trend_msg = "ERROR", "#888", "", "No Data"

    # 3. SECTOR LOGIC
    try:
        offense = (data['XLK'].pct_change(20).iloc[-1] + data['XLY'].pct_change(20).iloc[-1]) / 2
        defense = (data['XLU'].pct_change(20).iloc[-1] + data['XLP'].pct_change(20).iloc[-1]) / 2
        
        if offense > defense:
            sec_txt, sec_col, sec_css = "RISK ON", SUCCESS_COLOR, "status-green"
            sec_msg = "Cyclicals leading defensives."
        else:
            sec_txt, sec_col, sec_css = "RISK OFF", WARNING_COLOR, "status-red"
            sec_msg = "Defensives leading. Caution."
    except:
        sec_txt, sec_col, sec_css = "NEUTRAL", NEUTRAL_COLOR, "status-orange"
        sec_msg = "Sector data unclear."

    return {
        "vix": {"val": vix, "status": vix_txt, "color": vix_col, "css": vix_css, "msg": vix_msg},
        "trend": {"val": price, "ref": sma200, "status": trend_txt, "color": trend_col, "css": trend_css, "msg": trend_msg},
        "sector": {"status": sec_txt, "color": sec_col, "css": sec_css, "msg": sec_msg}
    }

# ---------------- VISUALIZATION HELPERS ----------------
def draw_mpl_gauge(val, color):
    """Draws a simple semi-circle gauge using Matplotlib"""
    fig, ax = plt.subplots(figsize=(4, 2))
    fig.patch.set_facecolor(CARD_COLOR)
    ax.set_facecolor(CARD_COLOR)
    
    # Draw Background Arc
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 0, 180, width=0.15, color='#333'))
    
    # Draw Value Arc (Simple logic: Map 0-40 VIX to 0-180 degrees)
    max_vix = 40
    angle_val = min(val, max_vix) / max_vix * 180
    ax.add_patch(patches.Wedge((0.5, 0), 0.4, 180 - angle_val, 180, width=0.15, color=color))
    
    # Add Text
    ax.text(0.5, 0, f"{val:.2f}", ha='center', va='bottom', fontsize=20, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.axis('off')
    plt.tight_layout()
    return fig

def draw_mpl_bar_chart(df):
    """Draws a horizontal bar chart for sectors"""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Colors based on value
    colors = [SUCCESS_COLOR if x >= 0 else WARNING_COLOR for x in df['Change']]
    
    bars = ax.barh(df['Sector'], df['Change'], color=colors, height=0.6)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='y', colors='white', labelsize=9)
    ax.tick_params(axis='x', colors='white', labelsize=8)
    ax.xaxis.grid(True, linestyle=':', alpha=0.3, color='#444')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + (0.1 if width > 0 else -0.1)
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', ha='left' if width > 0 else 'right', color='white', fontsize=8)
                
    ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    return fig

# ---------------- MAIN APP ----------------
try:
    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown("<h2 style='color:white; margin:0;'>DR CAPITAL</h2>", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='text-align: left; padding-top: 5px;'>
            <h1 style='margin:0; padding:0; font-size: 28px;'>Macro Command Center</h1>
            <p style='margin:0; font-size: 14px; color: gray;'>Put Credit Spread Environment Check</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0; border-top: 1px solid #333; margin: 15px 0;'>", unsafe_allow_html=True)

    # Fetch Data
    with st.spinner("Analyzing Market Structure..."):
        df, sector_tickers = get_macro_data()
        regime = analyze_regime(df)

    if not regime:
        st.error("Could not load market data. Please refresh.")
        st.stop()

    # ---------------- ROW 1: THE SIGNAL CARDS ----------------
    col1, col2, col3 = st.columns(3)

    # CARD 1: VIX
    with col1:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label">Volatility Regime (VIX)</div>
        """, unsafe_allow_html=True)
        # Use Matplotlib Gauge
        st.pyplot(draw_mpl_gauge(regime['vix']['val'], regime['vix']['color']), use_container_width=True)
        st.markdown(f"""
            <div class="macro-status {regime['vix']['css']}">{regime['vix']['status']}</div>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">{regime['vix']['msg']}</div>
        </div>
        """, unsafe_allow_html=True)

    # CARD 2: TREND
    with col2:
        trend_arrow = "▲" if regime['trend']['status'] == "BULLISH" else "▼"
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label">Market Trend (SPY)</div>
            <div class="macro-value" style="color: {regime['trend']['color']}">{trend_arrow} ${regime['trend']['val']:.2f}</div>
            <div class="macro-status {regime['trend']['css']}">{regime['trend']['status']}</div>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">
                vs 200 SMA: <span style="color: #ccc">${regime['trend']['ref']:.2f}</span><br>
                {regime['trend']['msg']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CARD 3: SECTOR
    with col3:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label">Sector Rotation</div>
            <div class="macro-value" style="color: {regime['sector']['color']}">{regime['sector']['status']}</div>
            <div style="font-size: 12px; color: #888; margin-bottom: 10px;">
                Funds flowing into: <br>
                <span style="color:white;">{ "Tech & Consumer" if regime['sector']['status'] == "RISK ON" else "Utilities & Staples" }</span>
            </div>
            <div class="macro-status {regime['sector']['css']}">{regime['sector']['status']}</div>
            <div style="font-size: 12px; color: #888; margin-top: 8px;">{regime['sector']['msg']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- ROW 2: SECTOR HEATMAP ----------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Where is the money going today?")
    
    # Calculate daily change
    daily_returns = df[sector_tickers].pct_change().iloc[-1] * 100
    sec_df = pd.DataFrame(daily_returns).reset_index()
    sec_df.columns = ['Sector', 'Change']
    sec_df = sec_df.sort_values('Change', ascending=True) # Ascending for horizontal bar chart
    
    # Draw Matplotlib Chart
    st.pyplot(draw_mpl_bar_chart(sec_df), use_container_width=True)

except Exception as e:
    st.error(f"Data Feed Error: {e}")
