import streamlit as st
from datetime import date, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from streamlit_autorefresh import st_autorefresh

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    log_completed_trade, 
    logout,
)

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

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

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

# Solid White Divider
st.markdown(WHITE_DIVIDER_HTML, unsafe_allow_html=True)

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

# --- Charting & Progress Bar Functions ---
def plot_spread_chart(df, trade_start_date, expiration_date, short_strike, long_strike, crit_price=None):
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

    if crit_price:
        ax.axhline(y=crit_price, color=STOP_LOSS_COLOR, linestyle=':', linewidth=1.2, label='Stop Loss (400%)')

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
    """
    Renders a Hybrid P&L Bar.
    - Positive (Green): % of Max Profit (Premium).
    - Negative (Red): % of Max Risk (Collateral).
    """
    if current_pl is None:
        return '<div style="color:gray; font-size:12px;">Pending P&L...</div>'
    
    # Calculate Hybrid %
    if current_pl >= 0:
        # Profit relative to Credit Collected (Target is 60%)
        display_pct = (current_pl / max_gain) * 100 if max_gain > 0 else 0
        is_profit = True
    else:
        # Loss relative to Max Risk (Stop is usually managed before -100%)
        display_pct = (current_pl / max_loss) * 100 if max_loss > 0 else 0
        is_profit = False

    # Visual Mapping for the CSS Bar (0 to 100 scale)
    # We want the bar to split in the middle (50%).
    # -100% Risk -> 0% Bar
    # 0% PnL -> 50% Bar
    # +60% Gain -> ~80% Bar (Target)
    # +100% Gain -> 100% Bar
    
    # Scale Logic:
    # If Profit: Map 0..100 to 50..100
    # If Loss: Map -100..0 to 0..50
    
    if is_profit:
        visual_fill = 50 + (display_pct / 2) # e.g., 60% profit -> 50 + 30 = 80% filled
    else:
        # display_pct is negative (e.g., -50%)
        # -100% -> 0 fill. -50% -> 25 fill. 0% -> 50 fill.
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

    # Target Marker Position (60% Profit = 80% visual position)
    target_marker_left = 80 

    return (
        f'<div style="margin-bottom: 12px; margin-top: 5px;">'
        f'<div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">'
        f'<strong style="color: #ddd;">Target Progress</strong>'
        f'<span style="color:{label_color}; font-weight:bold;">{status_text}</span>'
        f'</div>'
        
        f'<div style="width: 100%; background-color: #333; height: 8px; border-radius: 4px; position: relative; overflow: hidden; border: 1px solid #444;">'
            # The Fill Bar
            f'<div style="width: {visual_fill}%; background-color: {bar_color}; height: 100%; transition: width 0.5s ease-in-out;"></div>'
            
            # Break Even Marker (Center)
            f'<div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background-color: rgba(255,255,255,0.8);" title="Break Even"></div>'
            
            # Target Marker (60% Profit)
            f'<div style="position: absolute; left: {target_marker_left}%; top: 0; bottom: 0; width: 2px; background-color: {SUCCESS_COLOR}; opacity: 0.7;" title="Target (60%)"></div>'
        f'</div>'
        
        f'<div style="position: relative; height: 15px; font-size: 9px; color: gray; margin-top: 2px;">'
        f'<span style="position: absolute; left: 0;">Max Loss (-100%)</span>'
        f'<span style="position: absolute; left: 50%; transform: translateX(-50%);">Break Even</span>'
        f'<span style="position: absolute; left: {target_marker_left}%; transform: translateX(-50%);">Target (60%)</span>'
        f'</div>'
        f'</div>'
    )

# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Display Trades ----------------
if not st.session_state.trades:
    st.info("No active trades. Go to 'Spread Finder' to scan for new opportunities.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})
        current_dte = days_to_expiry(t["expiration"])
        
        # Ensure contracts is an int
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
                        f"<div style='background-color: rgba(0, 200, 83, 0.1); border: 1px solid {SUCCESS_COLOR}; color: {SUCCESS_COLOR}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; white-space: nowrap;'>"
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
            
            # UPDATED CALL: Pass current dollars, max loss, max gain
            st.markdown(render_profit_bar(pl_dollars, max_loss_total, max_gain_total), unsafe_allow_html=True)
            
            price_hist = t.get("cached", {}).get("price_history", [])
            stop_price = t.get("cached", {}).get("stop_loss_price")
            
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
                        long_strike=t['long_strike'],
                        crit_price=stop_price
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
