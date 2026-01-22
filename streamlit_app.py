import streamlit as st
from datetime import date, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from streamlit_autorefresh import st_autorefresh
import yfinance as yf 
import numpy as np # Added for safety checks

# ---------------- Persistence ----------------
try:
    from persistence import (
        ensure_logged_in,
        build_drive_service_from_session,
        save_to_drive,
        load_from_drive,
        log_completed_trade, 
        logout,
    )
except ImportError:
    st.error("Persistence module missing. Please check your file structure.")
    st.stop()

# ---------------- Page config ----------------
st.set_page_config(page_title="Dashboard", layout="wide")

# ---------------- Constants ----------------
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
STOP_LOSS_COLOR = "#FFA726"
WHITE_DIVIDER_HTML = "<hr style='border: 0; border-top: 1px solid #FFFFFF; margin-top: 10px; margin-bottom: 10px;'>"

# ---------------- UI Refresh ----------------
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
ensure_logged_in()

@st.cache_resource
def get_drive_connection():
    """Establishes the Drive connection only ONCE per session."""
    try:
        return build_drive_service_from_session()
    except Exception:
        return None

drive_service = get_drive_connection()

# ---------------- Header ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])

with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=130)
    except Exception:
        st.write("**DR CAPITAL**")

with header_col2:
    st.markdown("""
    <div style='text-align: left; padding-top: 10px;'>
        <h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Put Credit Spread Monitor</h1>
        <p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>
    </div>
    """, unsafe_allow_html=True)

with header_col3:
    st.write("") 
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
        st.rerun()

# ---------------- Helpers ----------------
def days_to_expiry(expiry) -> int:
    if isinstance(expiry, str):
        try:
            expiry = date.fromisoformat(expiry)
        except:
            return 0
    return max((expiry - date.today()).days, 0)

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

# ---------------- SNP Performance Widget ----------------
def render_snp_performance():
    """Fetches and displays S&P 500 daily performance."""
    try:
        snp = yf.Ticker("^GSPC")
        hist = snp.history(period="2d")
        
        if len(hist) < 2:
            return 
            
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2]
        change = current - prev
        pct_change = (change / prev) * 100
        
        color = SUCCESS_COLOR if change >= 0 else WARNING_COLOR
        arrow = "▲" if change >= 0 else "▼"
        
        st.markdown(f"""
        <div style="
            background-color: rgba(40, 40, 45, 0.6); 
            border: 1px solid #444; 
            border-radius: 6px; 
            padding: 8px 15px; 
            margin-bottom: 15px; 
            display: inline-flex; 
            align-items: center; 
            gap: 15px;">
            <span style="font-weight: bold; color: #fff; font-size: 14px;">S&P 500 (SPX)</span>
            <span style="font-size: 14px; color: #ddd;">${current:,.2f}</span>
            <span style="font-weight: bold; color: {color}; font-size: 14px;">{arrow} {pct_change:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass 

# --- Charting & Progress Bar Functions ---
def plot_spread_chart(df, trade_start_date, expiration_date, short_strike, long_strike):
    bg_color = '#0E1117'     
    card_color = '#262730'   
    text_color = '#FAFAFA'   
    grid_color = '#444444'   
    
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    width = 0.6    
    width2 = 0.05  
    
    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]
    
    ax.bar(up.Date, up.High - up.Low, bottom=up.Low, color=SUCCESS_COLOR, width=width2, align='center')
    ax.bar(down.Date, down.High - down.Low, bottom=down.Low, color=WARNING_COLOR, width=width2, align='center')
    ax.bar(up.Date, up.Close - up.Open, bottom=up.Open, color=SUCCESS_COLOR, width=width, align='center')
    ax.bar(down.Date, down.Open - down.Close, bottom=down.Close, color=WARNING_COLOR, width=width, align='center')

    min_date = df['Date'].min()
    max_date = max(df['Date'].max(), expiration_date) + pd.Timedelta(days=5)
    ax.set_xlim(left=min_date, right=max_date)

    ax.axvline(x=trade_start_date, color=SUCCESS_COLOR, linestyle='--', linewidth=1, label='Start', alpha=0.7)
    ax.axvline(x=expiration_date, color='#B0BEC5', linestyle='--', linewidth=1, label='Exp', alpha=0.7)
    
    warning_date = expiration_date - pd.Timedelta(days=21)
    ax.axvline(x=warning_date, color=STOP_LOSS_COLOR, linestyle='--', linewidth=1, label='21 Days Out', alpha=0.8)

    ax.axhline(y=short_strike, color='#FF5252', linestyle='-', linewidth=1.2, label='Strikes')
    ax.axhline(y=long_strike, color='#FF5252', linestyle='-', linewidth=1.2)
    ax.axhspan(short_strike, long_strike, color='#FF5252', alpha=0.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    
    ax.tick_params(axis='x', colors=text_color, labelsize=8)
    ax.tick_params(axis='y', colors=text_color, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(True, which='major', linestyle=':', color=grid_color, alpha=0.4)
    
    ax.set_title(f"Price Action vs Strikes (Exp: {expiration_date.date()})", 
                 color=text_color, fontsize=9, fontweight='bold', pad=10)
    
    leg = ax.legend(loc='upper left', fontsize=7, facecolor=card_color, edgecolor=grid_color)
    for text in leg.get_texts():
        text.set_color(text_color)

    plt.tight_layout()
    return fig

def render_profit_bar(current_pl, max_loss, max_gain):
    if current_pl is None:
        return '<div style="color:gray; font-size:12px;">Pending P&L...</div>'
    
    if current_pl >= 0:
        display_pct = (current_pl / max_gain) * 100 if max_gain > 0 else 0
        is_profit = True
    else:
        display_pct = (current_pl / max_loss) * 100 if max_loss > 0 else 0
        is_profit = False

    if is_profit:
        visual_fill = 50 + (display_pct / 2) 
    else:
        visual_fill = 50 - (abs(display_pct) / 2)
        
    visual_fill = max(0, min(visual_fill, 100))
    
    if is_profit:
        bar_color = SUCCESS_COLOR 
        label_color = SUCCESS_COLOR
        status_text = f"PROFIT: +{display_pct:.1f}% (of Credit)"
        if display_pct >= 60:
             status_text = f"WIN TARGET: {display_pct:.1f}%"
    else:
        bar_color = WARNING_COLOR
        label_color = WARNING_COLOR
        status_text = f"LOSS: {display_pct:.1f}% (of Risk)"

    target_marker_left = 80 

    html_block = f"""
<div style="margin-bottom: 12px; margin-top: 5px;">
<div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">
<strong style="color: #ddd;">Target Progress</strong>
<span style="color:{label_color}; font-weight:bold;">{status_text}</span>
</div>
<div style="width: 100%; background-color: #333; height: 8px; border-radius: 4px; position: relative; overflow: hidden; border: 1px solid #444;">
<div style="width: {visual_fill}%; background-color: {bar_color}; height: 100%; transition: width 0.5s ease-in-out;"></div>
<div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background-color: rgba(255,255,255,0.8);" title="Break Even"></div>
<div style="position: absolute; left: {target_marker_left}%; top: 0; bottom: 0; width: 2px; background-color: {SUCCESS_COLOR}; opacity: 0.7;" title="Target (60%)"></div>
</div>
<div style="position: relative; height: 15px; font-size: 9px; color: gray; margin-top: 2px;">
<span style="position: absolute; left: 0;">Max Loss (-100%)</span>
<span style="position: absolute; left: 50%; transform: translateX(-50%);">Break Even</span>
<span style="position: absolute; left: {target_marker_left}%; transform: translateX(-50%);">Target (60%)</span>
</div>
</div>
"""
    return html_block

def render_open_positions_grid(trades):
    if not trades:
        return

    style_block = """
<style>
.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 5px;
}
.mini-card {
    border-radius: 6px;
    padding: 10px;
    height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border: 1px solid #333;
    transition: transform 0.2s;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.mini-card:hover {
    border-color: #666;
    transform: translateY(-2px);
}
.mc-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0px;
}
.mc-ticker-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.mc-ticker {
    font-size: 16px;
    font-weight: bold;
    color: #fff;
}
.mc-pl-dollar {
    font-size: 15px;
    font-weight: bold;
    text-align: right;
}
.mc-body {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: auto;
}
.mc-row-spread {
    display: flex;
    justify-content: flex-end; 
    align-items: center;
    font-size: 13px;
    color: #ccc;
    line-height: 1.3;
}
.mc-row-std {
    display: flex;
    justify-content: space-between; 
    align-items: center;
    font-size: 13px;
    color: #ccc;
    line-height: 1.3;
}
.mc-val {
    font-weight: bold;
}
.sub-text {
    font-size: 10px; 
    color: #999; 
    font-weight: normal; 
    margin-left: 2px;
}
</style>
"""

    cards_html = ""

    for t in trades:
        cached = t.get("cached", {})
        ticker = t['ticker']
        contracts = int(t.get("contracts", 1)) 
        current_dte = days_to_expiry(t["expiration"])
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"] * 100 * contracts
        max_loss = (width - t["credit"]) * 100 * contracts
        
        profit_pct = cached.get("current_profit_percent")
        spread_value = cached.get("spread_value_percent", 0.0)
        day_change = cached.get("day_change_percent", 0.0)
        spy_crash_alert = cached.get("spy_below_ma", False)
        
        pl_dollars = 0.0
        if profit_pct is not None:
            pl_dollars = max_gain * (profit_pct / 100.0)

        # Status Logic
        card_status_msg = "Nominal"
        card_status_color = SUCCESS_COLOR

        if spy_crash_alert:
            card_status_msg = "CRASH ALERT"
            card_status_color = WARNING_COLOR
        elif profit_pct and profit_pct >= 60:
            card_status_msg = "TARGET HIT"
            card_status_color = SUCCESS_COLOR
        else:
            if spread_value and spread_value >= 400:
                card_status_msg = "STOP LOSS"
                card_status_color = WARNING_COLOR
            elif current_dte <= 21:
                card_status_msg = "< 21 DTE"
                card_status_color = WARNING_COLOR
            else:
                card_status_msg = "Nominal"
                card_status_color = SUCCESS_COLOR

        # Color Logic
        bg_color = "rgba(40, 40, 45, 0.8)" 
        border_color = "#333"

        pl_str = f"{'+' if pl_dollars > 0 else ''}${pl_dollars:.0f}"
        pl_dollar_color = "#ccc"

        # --- SAFE PROFIT DISPLAY FIX ---
        # This handles the case where profit_pct is None (New Trade)
        if pl_dollars >= 0:
            if profit_pct is not None:
                ratio = min(profit_pct / 60.0, 1.0)
                alpha = 0.1 + (ratio * 0.2) 
                pl_text = f"+{profit_pct:.1f}%"
            else:
                ratio = 0
                alpha = 0.1
                pl_text = "0.0%"
            
            bg_color = f"rgba(0, 200, 83, {alpha})"
            pl_color = SUCCESS_COLOR
            pl_dollar_color = SUCCESS_COLOR
        else:
            loss_pct_of_risk = (pl_dollars / max_loss) * 100 if max_loss > 0 else 0
            ratio = min(abs(loss_pct_of_risk) / 100.0, 1.0) 
            alpha = 0.1 + (ratio * 0.2)
            bg_color = f"rgba(211, 47, 47, {alpha})"
            pl_text = f"{loss_pct_of_risk:.1f}%"
            pl_color = "#ff6b6b"
            pl_dollar_color = "#ff6b6b"

        # Safe Spread Value
        if spread_value is not None:
            spread_val_str = f"{spread_value:.0f}%"
            if spread_value >= 400:
                spread_color = WARNING_COLOR
                border_color = WARNING_COLOR
                bg_color = "rgba(100, 0, 0, 0.3)"
            else:
                spread_color = SUCCESS_COLOR
        else:
            spread_val_str = "-"
            spread_color = "#ccc"

        if day_change is None: day_change = 0.0
        if day_change > 0:
            day_fmt = f"<span style='color:{SUCCESS_COLOR}; font-size:12px;'>▲{day_change:.1f}%</span>"
        elif day_change < 0:
            day_fmt = f"<span style='color:{WARNING_COLOR}; font-size:12px;'>▼{abs(day_change):.1f}%</span>"
        else:
            day_fmt = f"<span style='color:gray; font-size:12px;'>0.0%</span>"

        cards_html += f"""
<div class="mini-card" style="background-color: {bg_color}; border-color: {border_color};">
<div class="mc-header">
<div class="mc-ticker-row">
<div class="mc-ticker">{ticker}</div>
{day_fmt}
</div>
<div class="mc-pl-dollar" style="color:{pl_dollar_color};">{pl_str}</div>
</div>
<div class="mc-body">
<div class="mc-row-spread">
<div style="text-align:right;">
<span style="font-size:11px; color:#aaa; margin-right:4px;">Spread Value:</span>
<span class="mc-val" style="color:{spread_color};">{spread_val_str}</span>
<span class="sub-text" style="display:block;">(Must not exceed 400%)</span>
</div>
</div>
<div class="mc-row-std">
<span>P&L:</span>
<span class="mc-val" style="color:{pl_color};">{pl_text}</span>
</div>
<div class="mc-row-std">
<span>Status:</span>
<span class="mc-val" style="color:{card_status_color};">{card_status_msg}</span>
</div>
</div>
</div>"""

    final_html = f"""
{style_block}
<h3 style='margin-bottom: 15px; font-size: 18px; border: 0; padding-bottom: 8px;'>Open Positions</h3>
<div class="grid-container">
{cards_html}
</div>
"""
    st.markdown(final_html, unsafe_allow_html=True)


# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Open Positions Grid ----------------
if st.session_state.trades:
    render_snp_performance() 
    render_open_positions_grid(st.session_state.trades)
    st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

# ---------------- Display Detailed Trades ----------------
if not st.session_state.trades:
    render_snp_performance()
    st.info("No active trades. Go to 'Spread Finder' to scan for new opportunities.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})
        current_dte = days_to_expiry(t["expiration"])
        
        contracts = int(t.get("contracts", 1)) 
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain_total = t["credit"] * 100 * contracts
        max_loss_total = (width - t["credit"]) * 100 * contracts

        current_price = cached.get("current_price")
        
        abs_delta = cached.get("abs_delta") 
        if abs_delta is None and cached.get("delta"): 
             abs_delta = abs(cached.get("delta"))
        
        net_theta = cached.get("net_theta", 0.0)
        daily_theta_dollars = net_theta * 100.0 * contracts 

        pop_percent = 0.0
        if abs_delta is not None:
            pop_percent = (1.0 - abs_delta) * 100.0

        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        
        spy_crash_alert = cached.get("spy_below_ma", False) 

        # --- Status Logic ---
        status_msg = "Status Nominal"
        status_icon = "✅"
        status_color = SUCCESS_COLOR

        if spy_crash_alert:
            status_icon = "🚨"
            status_msg = "MARKET CRASH ALERT (SPY < 200 SMA)"
            status_color = WARNING_COLOR
        elif profit_pct and profit_pct >= 60:
            status_icon = "💰" 
            status_msg = "TARGET REACHED (60%)"
            status_color = SUCCESS_COLOR
        else:
            if spread_value and spread_value >= 400:
                status_icon = "⚠️"
                status_color = WARNING_COLOR
                status_msg = "Stop Loss Hit (>400%)"
            elif current_dte <= 21:
                status_icon = "⚠️"
                status_color = WARNING_COLOR
                status_msg = "Exit Zone (<21 DTE)"
        
        spread_color = WARNING_COLOR if spread_value and spread_value >= 400 else SUCCESS_COLOR
        spread_val = f"{spread_value:.0f}" if spread_value is not None else "Pending"
        
        dte_color = WARNING_COLOR if current_dte <= 21 else SUCCESS_COLOR
        
        pop_color = SUCCESS_COLOR if pop_percent >= 60 else "#FFA726"
        if pop_percent < 50: pop_color = WARNING_COLOR
        
        spy_status_text = "BULLISH (Index above 200 SMA)"
        spy_status_color = SUCCESS_COLOR
        if spy_crash_alert:
            spy_status_text = "BEARISH (Index bellow 200 SMA)"
            spy_status_color = WARNING_COLOR

        cols = st.columns([3, 4])

        # -------- LEFT CARD --------
        with cols[0]:
            day_change = cached.get("day_change_percent", 0.0)
            if day_change is None: day_change = 0.0
            
            if day_change > 0:
                change_color = SUCCESS_COLOR 
                arrow = "▲"
                change_str = f"{day_change:.2f}%"
            elif day_change < 0:
                change_color = WARNING_COLOR
                arrow = "▼"
                change_str = f"{abs(day_change):.2f}%" 
            else:
                change_color = "gray"
                arrow = ""
                change_str = "0.00%"

            price_display = f"${current_price:.2f}" if current_price else "$-.--"
            
            # --- UPDATED THETA LOGIC ---
            theta_box_bg = "rgba(0, 200, 83, 0.1)"
            theta_box_border = SUCCESS_COLOR
            theta_box_text_color = SUCCESS_COLOR
            
            is_negative_theta_risk = False
            if current_price is not None and current_price < t['short_strike'] and daily_theta_dollars < 0:
                is_negative_theta_risk = True
                theta_box_bg = "rgba(211, 47, 47, 0.1)"
                theta_box_border = WARNING_COLOR
                theta_box_text_color = WARNING_COLOR

            theta_text = f"+${daily_theta_dollars:.2f} Today" if daily_theta_dollars >= 0 else f"-${abs(daily_theta_dollars):.2f} Today"
            
            left_card_html = (
                f"<div style='line-height: 1.4; font-size: 15px;'>"
                f"<div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px;'>"
                    f"<h3 style='margin: 0; display: flex; align-items: center; gap: 8px;'>"
                    f"{t['ticker']} "
                    f"<span style='font-size: 0.9em; color: #ddd; font-weight: normal;'>{price_display}</span>"
                    f"<span style='color: {change_color}; font-size: 0.85em;'>"
                    f"{arrow} {change_str}"
                    f"</span>"
                    f"</h3>"
                    f"<div style='display:flex; flex-direction:column; align-items:flex-end; gap:2px;'>"
                        f"<span style='font-size:10px; color:gray; text-transform:uppercase; letter-spacing:0.5px;'>Daily Theta Gain</span>"
                        f"<div style='background-color: {theta_box_bg}; border: 1px solid {theta_box_border}; color: {theta_box_text_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; white-space: nowrap;'>"
                        f"{theta_text}"
                        f"</div>"
                    f"</div>"
                f"</div>"
                
                f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2px;'>"
                    f"<div><strong>Short:</strong> {t['short_strike']}</div>"
                    f"<div><strong>Max Gain:</strong> {format_money(max_gain_total)}</div>"
                    f"<div><strong>Long:</strong> {t['long_strike']}</div>"
                    f"<div><strong>Max Loss:</strong> {format_money(max_loss_total)}</div>"
                    f"<div><strong>Width:</strong> {width:.2f}</div>"
                    f"<div><strong>Contracts:</strong> {contracts}</div>"
                    f"<div style='grid-column: span 2;'><strong>Exp:</strong> {t['expiration']}</div>"
                f"</div>"
                
                f"<div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid #444; display: flex; justify-content: space-between; align-items: center;'>"
                    f"<div style='color: {status_color}; font-weight: bold;'>"
                    f"{status_icon} {status_msg}"
                    f"</div>"
                    f"<div style='font-size: 13px; color: gray;'>"
                    f"P.O.P: <strong style='color: {pop_color};'>{pop_percent:.0f}%</strong>"
                    f"</div>"
                f"</div>"
                f"</div>"
            )
            st.markdown(left_card_html, unsafe_allow_html=True)
            
            st.write("") 
            
            if st.button("Close Position / Log", key=f"btn_close_{i}"):
                st.session_state[f"close_mode_{i}"] = True

            if st.session_state.get(f"close_mode_{i}", False):
                with st.container():
                    st.markdown("---")
                    st.info("📉 Closing Position & Logging to Journal")
                    with st.form(key=f"close_form_{i}"):
                        col_log1, col_log2 = st.columns(2)
                        with col_log1:
                            default_debit = 0.0
                            current_short = t.get("cached", {}).get("short_option_price")
                            current_long = t.get("cached", {}).get("long_option_price")
                            if current_short is not None and current_long is not None:
                                est_price = current_short - current_long
                                if est_price > 0:
                                    default_debit = est_price
                            
                            debit_paid = st.number_input("Debit Paid ($)", min_value=0.0, value=float(f"{default_debit:.2f}"), step=0.01)
                            exit_date_val = st.date_input("Exit Date", value=datetime.now().date())
                        
                        with col_log2:
                            close_notes = st.text_area("Notes", height=70)
                        
                        if st.form_submit_button("Confirm Close"):
                            if drive_service:
                                trade_data = t.copy()
                                trade_data['debit_paid'] = debit_paid
                                trade_data['notes'] = close_notes
                                trade_data['exit_date'] = exit_date_val.isoformat()
                                trade_data['contracts'] = contracts
                                
                                if log_completed_trade(drive_service, trade_data):
                                    st.success(f"Logged {t['ticker']}")
                                    st.session_state.trades.pop(i)
                                    save_to_drive(drive_service, st.session_state.trades)
                                    del st.session_state[f"close_mode_{i}"]
                                    st.rerun()
                                else:
                                    st.error("Drive Error: Could not log to sheet.")
                            else:
                                st.session_state.trades.pop(i)
                                del st.session_state[f"close_mode_{i}"]
                                st.rerun()

                    if st.button("Cancel", key=f"cancel_{i}"):
                        del st.session_state[f"close_mode_{i}"]
                        st.rerun()

        # -------- RIGHT CARD --------
        with cols[1]:
            pl_dollars = 0.0
            if profit_pct is not None:
                pl_dollars = max_gain_total * (profit_pct / 100.0)
            
            pl_text_color = SUCCESS_COLOR if pl_dollars >= 0 else WARNING_COLOR
            pl_str = f"{'+' if pl_dollars > 0 else ''}${pl_dollars:.2f}"

            right_card_html = (
                f"<div style='font-size: 14px; margin-bottom: 5px; position: relative;'>"
                f"<div style='position: absolute; top: 0; right: 0; font-weight: bold; color: {pl_text_color}; font-size: 1.1em;'>{pl_str}</div>"
                f"<div style='margin-bottom: 4px; padding-right: 70px;'>Market Regime: <strong style='color:{spy_status_color}'>{spy_status_text}</strong></div>"
                f"<div style='margin-bottom: 4px;'>Spread Value: <strong style='color:{spread_color}'>{spread_val}%</strong> <span style='color:gray; font-size:0.85em;'>(Must not exceed 400%)</span></div>"
                f"<div>DTE: <strong style='color:{dte_color}'>{current_dte}</strong> <span style='color:gray; font-size:0.85em;'>(Must be greater than 21 days)</span></div>"
                f"</div>"
            )
            st.markdown(right_card_html, unsafe_allow_html=True)
            
            st.markdown(render_profit_bar(pl_dollars, max_loss_total, max_gain_total), unsafe_allow_html=True)
            
            price_hist = t.get("cached", {}).get("price_history", [])
            
            if price_hist:
                try:
                    df_chart = pd.DataFrame(price_hist)
                    df_chart['Date'] = pd.to_datetime(df_chart['date'])
                    df_chart['Close'] = df_chart['close']
                    df_chart['Open'] = df_chart['open'] if 'open' in df_chart.columns else df_chart['close']
                    df_chart['High'] = df_chart['high'] if 'high' in df_chart.columns else df_chart['close']
                    df_chart['Low'] = df_chart['low'] if 'low' in df_chart.columns else df_chart['close']
                    
                    fig = plot_spread_chart(
                        df=df_chart,
                        trade_start_date=pd.Timestamp(t['entry_date']),
                        expiration_date=pd.Timestamp(t['expiration']),
                        short_strike=t['short_strike'],
                        long_strike=t['long_strike']
                    )
                    st.pyplot(fig)
                except Exception:
                    st.caption("Chart Error")
            else:
                st.caption("Loading chart...")

        st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

# ---------------- External Tools ----------------
t1, t2 = st.columns(2)
with t1: st.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
with t2: st.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
