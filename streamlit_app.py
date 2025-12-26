import streamlit as st
from datetime import date, datetime
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textwrap

# ---------------- Persistence ----------------
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    log_trade_to_csv, 
    logout,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")

# ---------------- Constants ----------------
SUCCESS_COLOR = "#00C853"  # Unified Green
WARNING_COLOR = "#d32f2f"  # Red
STOP_LOSS_COLOR = "#FFA726" # Orange

# ---------------- UI Refresh ----------------
st_autorefresh(interval=60_000, key="ui_refresh")

# ---------------- Auth / Drive ----------------
try:
    ensure_logged_in()
except Exception:
    st.warning("Google OAuth configured.")

drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# ---------------- Header & Logo ----------------
header_col1, header_col2, header_col3 = st.columns([1.5, 7, 1.5])

with header_col1:
    try:
        st.image("754D6DFF-2326-4C87-BB7E-21411B2F2373.PNG", width=140)
    except Exception:
        st.write("**DR CAPITAL**")

with header_col2:
    header_html = (
        "<div style='text-align: left; padding-top: 10px;'>"
        "<h1 style='margin-bottom: 0px; padding-bottom: 0px;'>Put Credit Spread Monitor</h1>"
        "<p style='margin-top: 0px; font-size: 18px; color: gray;'>Strategic Options Management System</p>"
        "</div>"
    )
    st.markdown(header_html, unsafe_allow_html=True)

with header_col3:
    st.write("") 
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            st.session_state.pop("credentials", None)
        st.experimental_rerun()

st.markdown("---")

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

def get_entry_dte(entry_date_str, expiry_date_str):
    try:
        entry = date.fromisoformat(entry_date_str)
        expiry = date.fromisoformat(expiry_date_str)
        return (expiry - entry).days
    except:
        return 30 # fallback

# --- 1. CHARTING FUNCTION ---
def plot_spread_chart(df, trade_start_date, expiration_date, short_strike, long_strike, crit_price=None):
    """
    Generates a Dark Mode Matplotlib figure with Candlesticks.
    """
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
    
    col_up = '#26a69a'
    col_down = '#ef5350'
    
    # Draw Wicks
    ax.bar(up.Date, up.High - up.Low, bottom=up.Low, color=col_up, width=width2, align='center')
    ax.bar(down.Date, down.High - down.Low, bottom=down.Low, color=col_down, width=width2, align='center')
    
    # Draw Bodies
    ax.bar(up.Date, up.Close - up.Open, bottom=up.Open, color=col_up, width=width, align='center')
    ax.bar(down.Date, down.Open - down.Close, bottom=down.Close, color=col_down, width=width, align='center')

    # Timeline
    min_date = df['Date'].min()
    max_date = max(df['Date'].max(), expiration_date) + pd.Timedelta(days=5)
    ax.set_xlim(left=min_date, right=max_date)

    # Event Lines
    ax.axvline(x=trade_start_date, color=SUCCESS_COLOR, linestyle='--', linewidth=1, label='Start', alpha=0.7)
    ax.axvline(x=expiration_date, color='#B0BEC5', linestyle='--', linewidth=1, label='Exp', alpha=0.7)

    # Strikes
    ax.axhline(y=short_strike, color='#FF5252', linestyle='-', linewidth=1.2, label='Strikes')
    ax.axhline(y=long_strike, color='#FF5252', linestyle='-', linewidth=1.2)
    ax.axhspan(short_strike, long_strike, color='#FF5252', alpha=0.15)

    if crit_price:
        ax.axhline(y=crit_price, color=STOP_LOSS_COLOR, linestyle=':', linewidth=1.2, label='Stop Loss')

    # Formatting
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

# --- 2. PROGRESS BAR FUNCTION ---
def render_profit_bar(profit_pct):
    if profit_pct is None:
        return '<div style="color:gray; font-size:12px;">Pending P&L...</div>'
    
    fill_pct = ((profit_pct + 100) / 150) * 100
    display_fill = max(0, min(fill_pct, 100))
    
    if profit_pct < 0:
        bar_color = WARNING_COLOR 
        label_color = WARNING_COLOR
        status_text = f"LOSS: {profit_pct:.1f}%"
    elif profit_pct < 50:
        bar_color = SUCCESS_COLOR
        label_color = SUCCESS_COLOR
        status_text = f"PROFIT: {profit_pct:.1f}%"
    else:
        bar_color = SUCCESS_COLOR 
        label_color = SUCCESS_COLOR
        status_text = f"WIN TARGET: {profit_pct:.1f}%"

    return (
        f'<div style="margin-bottom: 12px; margin-top: 5px;">'
        f'<div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">'
        f'<strong style="color: #ddd;">Target Progress</strong>'
        f'<span style="color:{label_color}; font-weight:bold;">{status_text}</span>'
        f'</div>'
        f'<div style="width: 100%; background-color: #333; height: 6px; border-radius: 3px; position: relative; overflow: hidden; border: 1px solid #444;">'
        f'<div style="width: {display_fill}%; background-color: {bar_color}; height: 100%; transition: width 0.5s ease-in-out;"></div>'
        f'<div style="position: absolute; left: 66.6%; top: 0; bottom: 0; width: 1px; background-color: rgba(255,255,255,0.5);" title="Break Even (0%)"></div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; font-size:9px; color:gray; margin-top:2px; padding-left: 2px; padding-right: 2px;">'
        f'<span>Max Loss</span>'
        f'<span style="margin-left: 15px;">Break Even</span>'
        f'<span>TARGET (50%)</span>'
        f'</div>'
        f'</div>'
    )

# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Add Trade ----------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("New Position Entry")

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        ticker = st.text_input("Ticker").upper()
        num_contracts = st.number_input("Contracts", min_value=1, value=1, step=1)
        
    with c2:
        short_strike = st.number_input("Short Strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long Strike", min_value=0.0, format="%.2f")
        
    with c3:
        expiration = st.date_input("Expiration Date")
        entry_date = st.date_input("Entry Date")
        
    with c4:
        credit = st.number_input("Credit (Per Share)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Initialize Position")

    if submitted:
        if not ticker:
            st.warning("Ticker required.")
        elif long_strike >= short_strike:
            st.warning("Long strike must be lower than short strike.")
        else:
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "contracts": num_contracts, 
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration.isoformat(),
                "credit": credit,
                "entry_date": entry_date.isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "cached": {},
                "pnl_history": []
            }
            st.session_state.trades.append(trade)
            if drive_service:
                save_to_drive(drive_service, st.session_state.trades)
            st.success(f"Position initialized: {num_contracts}x {ticker} Puts.")

st.markdown("---")

# ---------------- Display Trades ----------------
st.subheader("Active Portfolio")

if not st.session_state.trades:
    st.info("No active trades.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})

        current_dte = days_to_expiry(t["expiration"])
        
        # --- POSITION SIZING & TOTALS ---
        contracts = t.get("contracts", 1) 
        
        width = abs(t["short_strike"] - t["long_strike"])
        
        # Max Gain = Credit * 100 * Contracts
        max_gain_total = t["credit"] * 100 * contracts
        
        # Max Loss = (Width - Credit) * 100 * Contracts
        max_loss_total = (width - t["credit"]) * 100 * contracts

        # Backend Data
        abs_delta = cached.get("abs_delta") 
        if abs_delta is None and cached.get("delta"): 
             abs_delta = abs(cached.get("delta"))
        
        # --- THETA CALCULATION ---
        net_theta = cached.get("net_theta", 0.0)
        daily_theta_dollars = net_theta * 100.0 * contracts 

        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        # --- Status Logic ---
        status_msg = "Status Nominal"
        status_icon = "✅"
        status_color = SUCCESS_COLOR

        if rules.get("other_rules", False):
            status_icon = "⚠️"
            status_color = WARNING_COLOR 
            if abs_delta and abs_delta >= 0.40:
                status_msg = "Short Delta High"
            elif spread_value and spread_value >= 150:
                status_msg = "Spread Value High"
            elif current_dte <= 7:
                status_msg = "Expiration Imminent"
        
        if profit_pct and profit_pct >= 50:
            status_icon = "💰" 
            status_msg = "TARGET REACHED"
            status_color = SUCCESS_COLOR

        # --- Color Coding ---
        delta_color = WARNING_COLOR if abs_delta and abs_delta >= 0.40 else SUCCESS_COLOR
        delta_val = f"{abs_delta:.2f}" if abs_delta is not None else "Pending"

        spread_color = WARNING_COLOR if spread_value and spread_value >= 150 else SUCCESS_COLOR
        spread_val = f"{spread_value:.0f}" if spread_value is not None else "Pending"

        dte_color = WARNING_COLOR if current_dte <= 7 else SUCCESS_COLOR

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

            theta_text = f"+${daily_theta_dollars:.2f} Today" if daily_theta_dollars >= 0 else f"-${abs(daily_theta_dollars):.2f} Today"
            
            # --- CARD HTML UPDATE ---
            left_card_html = (
                f"<div style='line-height: 1.4; font-size: 15px;'>"
                # HEADER ROW (Ticker + Theta)
                f"<div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px;'>"
                    f"<h3 style='margin: 0; display: flex; align-items: center; gap: 10px;'>"
                    f"{t['ticker']} "
                    f"<span style='color: {change_color}; font-size: 0.85em;'>"
                    f"{arrow} {change_str}"
                    f"</span>"
                    f"</h3>"
                    # Right side: Label + Badge
                    f"<div style='display:flex; flex-direction:column; align-items:flex-end; gap:2px;'>"
                        f"<span style='font-size:10px; color:gray; text-transform:uppercase; letter-spacing:0.5px;'>Daily Theta Gain</span>"
                        f"<div style='background-color: rgba(0, 200, 83, 0.1); border: 1px solid {SUCCESS_COLOR}; color: {SUCCESS_COLOR}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; white-space: nowrap;'>"
                        f"{theta_text}"
                        f"</div>"
                    f"</div>"
                f"</div>"
                
                # DATA GRID
                f"<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 2px;'>"
                    f"<div><strong>Short:</strong> {t['short_strike']}</div>"
                    f"<div><strong>Max Gain:</strong> {format_money(max_gain_total)}</div>"
                    
                    f"<div><strong>Long:</strong> {t['long_strike']}</div>"
                    f"<div><strong>Max Loss:</strong> {format_money(max_loss_total)}</div>"
                    
                    f"<div><strong>Width:</strong> {width:.2f}</div>"
                    f"<div><strong>Contracts:</strong> {contracts}</div>"
                    
                    f"<div style='grid-column: span 2;'><strong>Exp:</strong> {t['expiration']}</div>"
                f"</div>"
                
                # STATUS FOOTER
                f"<div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; color: {status_color}; font-weight: bold;'>"
                f"{status_icon} {status_msg}"
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
                        with col_log2:
                            close_notes = st.text_area("Notes", height=70)
                        
                        if st.form_submit_button("Confirm Close"):
                            if drive_service:
                                if log_trade_to_csv(drive_service, t, debit_paid, close_notes):
                                    st.success(f"Logged {t['ticker']}")
                                    st.session_state.trades.pop(i)
                                    save_to_drive(drive_service, st.session_state.trades)
                                    del st.session_state[f"close_mode_{i}"]
                                    st.experimental_rerun()
                                else:
                                    st.error("Drive Error")
                            else:
                                st.session_state.trades.pop(i)
                                del st.session_state[f"close_mode_{i}"]
                                st.experimental_rerun()
                    if st.button("Cancel", key=f"cancel_{i}"):
                        del st.session_state[f"close_mode_{i}"]
                        st.experimental_rerun()

        # -------- RIGHT CARD --------
        with cols[1]:
            right_card_html = (
                f"<div style='font-size: 14px; margin-bottom: 5px;'>"
                f"<div style='margin-bottom: 4px;'>Short-delta: <strong style='color:{delta_color}'>{delta_val}</strong> <span style='color:gray; font-size:0.85em;'>(Limit: 0.40)</span></div>"
                f"<div style='margin-bottom: 4px;'>Spread Value: <strong style='color:{spread_color}'>{spread_val}%</strong> <span style='color:gray; font-size:0.85em;'>(Limit: 150%)</span></div>"
                f"<div>DTE: <strong style='color:{dte_color}'>{current_dte}</strong> <span style='color:gray; font-size:0.85em;'>(Min: 7 days)</span></div>"
                f"</div>"
            )
            st.markdown(right_card_html, unsafe_allow_html=True)
            
            st.markdown(render_profit_bar(profit_pct), unsafe_allow_html=True)
            
            # Chart
            price_hist = t.get("cached", {}).get("price_history", [])
            crit_price = t.get("cached", {}).get("critical_price_040")
            
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
                        crit_price=crit_price
                    )
                    st.pyplot(fig)
                except Exception:
                    st.caption("Chart Error")
            else:
                st.caption("Loading chart...")

        st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# ---------------- Manual Controls ----------------
st.write("### Data Sync")
ctl1, ctl2, ctl_spacer = st.columns([1.5, 1.5, 5])
with ctl1:
    if st.button("💾 Save to Drive"):
        if drive_service and save_to_drive(drive_service, st.session_state.trades):
            st.success("Saved.")
with ctl2:
    if st.button("📥 Reload from Drive"):
        if drive_service:
            loaded = load_from_drive(drive_service)
            if loaded is not None:
                st.session_state.trades = loaded
                st.experimental_rerun()

st.markdown("---")

# ---------------- External Tools ----------------
st.subheader("External Tools")
t1, t2, t3, t4 = st.columns(4)
with t1: st.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
with t2: st.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
with t3: st.link_button("Option Screener", "https://optionmoves.com/", use_container_width=True)
with t4: st.link_button("IV Rank", "https://marketchameleon.com/", use_container_width=True)
