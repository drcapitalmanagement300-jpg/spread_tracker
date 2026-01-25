import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------- CONFIGURATION ----------------
st.set_page_config(layout="wide", page_title="Macro Command Center", page_icon="⚡")

# ---------------- CONSTANTS & STYLE ----------------
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
NEUTRAL_COLOR = "#FFA726"
BG_COLOR = "#0E1117"
CARD_COLOR = "#262730"

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
    data = yf.download(all_ticks, period="6mo", interval="1d", progress=False)['Close']
    return data, tickers['Sectors']

def analyze_regime(data):
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
    spy = data['SPY']
    sma200 = spy.rolling(200).mean().iloc[-1]
    price = spy.iloc[-1]
    
    if price > sma200:
        trend_txt, trend_col, trend_css = "BULLISH", SUCCESS_COLOR, "status-green"
        trend_msg = "SPY > 200 SMA. Safe to sell puts."
    else:
        trend_txt, trend_col, trend_css = "BEARISH", WARNING_COLOR, "status-red"
        trend_msg = "SPY < 200 SMA. Cash is king."

    # 3. SECTOR LOGIC (Risk On/Off)
    # Offense: Tech (XLK) + Discretionary (XLY)
    # Defense: Utilities (XLU) + Staples (XLP)
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

# ---------------- UI COMPONENTS ----------------
def draw_gauge(val, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        number = {'font': {'color': "white", 'size': 24}},
        gauge = {
            'axis': {'range': [0, 40], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 13], 'color': 'rgba(211, 47, 47, 0.2)'},
                {'range': [13, 20], 'color': 'rgba(0, 200, 83, 0.2)'},
                {'range': [20, 30], 'color': 'rgba(255, 167, 38, 0.2)'},
                {'range': [30, 50], 'color': 'rgba(211, 47, 47, 0.2)'}
            ],
        }))
    fig.update_layout(height=120, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

# ---------------- MAIN APP ----------------
try:
    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        # Use simple text if image not available on cloud
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

    # ---------------- ROW 1: THE SIGNAL CARDS ----------------
    col1, col2, col3 = st.columns(3)

    # CARD 1: VIX
    with col1:
        st.markdown(f"""
        <div class="macro-card">
            <div class="macro-label">Volatility Regime (VIX)</div>
            <div style="height: 120px;">
        """, unsafe_allow_html=True)
        st.plotly_chart(draw_gauge(regime['vix']['val'], regime['vix']['color']), use_container_width=True)
        st.markdown(f"""
            </div>
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
    sec_df = sec_df.sort_values('Change', ascending=False)
    
    # Custom Bar Chart matching UI
    fig_sec = px.bar(
        sec_df, x='Sector', y='Change',
        color='Change',
        color_continuous_scale=[WARNING_COLOR, "#333", SUCCESS_COLOR],
        range_color=[-2, 2],
        text_auto='.2f'
    )
    fig_sec.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font={'color': 'white'},
        xaxis={'title': None, 'gridcolor': '#333'},
        yaxis={'title': '% Change', 'gridcolor': '#333'},
        coloraxis_showscale=False,
        height=300
    )
    fig_sec.update_traces(textposition='outside')
    st.plotly_chart(fig_sec, use_container_width=True)

except Exception as e:
    st.error(f"Data Feed Error: {e}")
