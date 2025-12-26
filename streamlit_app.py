import streamlit as st
from datetime import date, datetime
import pandas as pd
import altair as alt
from streamlit_autorefresh import st_autorefresh
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# --- NEW CHARTING FUNCTION (Dark Mode + Candles) ---
def plot_spread_chart(df, trade_start_date, expiration_date, short_strike, long_strike, crit_price=None):
    """
    Generates a Dark Mode Matplotlib figure with Candlesticks.
    """
    # 1. Colors & Theme
    bg_color = '#0E1117'    # Streamlit Main Dark BG
    card_color = '#262730'  # Streamlit Card BG
    text_color = '#FAFAFA'  # White/Light Text
    grid_color = '#444444'  
    
    # 2. Setup Figure (Smaller Size)
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Apply Dark Background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # 3. Plot Candlesticks
    # We iterate manually to draw candles using Bar/Vlines logic
    width = 0.6  # width of candle body
    width2 = 0.05 # width of wick
    
    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]
    
    # Up Candles (Green)
    col_up = '#26a69a'
    col_down = '#ef5350'
    
    # Wicks
    ax.bar(up.Date, up.High - up.Low, bottom=up.Low, color=col_up, width=width2, align='center')
    ax.bar(down.Date, down.High - down.Low, bottom=down.Low, color=col_down, width=width2, align='center')
    
    # Bodies
    ax.bar(up.Date, up.Close - up.Open, bottom=up.Open, color=col_up, width=width, align='center')
    ax.bar(down.Date, down.Open - down.Close, bottom=down.Close, color=col_down, width=width, align='center')

    # 4. Forward Space & X-Axis Limits
    min_date = df['Date'].min()
    max_date = max(df['Date'].max(), expiration_date) + pd.Timedelta(days=5)
    ax.set_xlim(left=min_date, right=max_date)

    # 5. Vertical Event Lines
    ax.axvline(x=trade_start_date, color='#00C853', linestyle='--', linewidth=1, label='Start', alpha=0.7)
    ax.axvline(x=expiration_date, color='#B0BEC5', linestyle='--', linewidth=1, label='Exp', alpha=0.7)

    # 6. Unified Strikes & Shading
    # Strikes (Red)
    ax.axhline(y=short_strike, color='#FF5252', linestyle='-', linewidth=1.2, label='Strikes')
    ax.axhline(y=long_strike, color='#FF5252', linestyle='-', linewidth=1.2)
    
    # Shaded Spread Zone
    ax.axhspan(short_strike, long_strike, color='#FF5252', alpha=0.15)

    # 7. Dynamic Critical Price (Stop Loss)
    if crit_price:
        ax.axhline(y=crit_price, color='#FFA726', linestyle=':', linewidth=1.2, label='Stop Loss')

    # 8. Styling & Legend
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)
    
    # Ticks and Labels
    ax.tick_params(axis='x', colors=text_color, labelsize=8)
    ax.tick_params(axis='y', colors=text_color, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    # Grid
    ax.grid(True, which='major', linestyle=':', color=grid_color, alpha=0.4)
    
    # Title
    ax.set_title(f"Price Action vs Strikes (Exp: {expiration_date.date()})", 
                 color=text_color, fontsize=9, fontweight='bold', pad=10)
    
    # Legend (Dark Mode Adaptation)
    leg = ax.legend(loc='upper left', fontsize=7, facecolor=card_color, edgecolor=grid_color)
    for text in leg.get_texts():
        text.set_color(text_color)

    plt.tight_layout()
    return fig

# ---------------- Load Drive State ----------------
if drive_service:
    st.session_state.trades = load_from_drive(drive_service) or []
else:
    if "trades" not in st.session_state:
        st.session_state.trades = []

# ---------------- Add Trade ----------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("New Position Entry")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Ticker").upper()
        short_strike = st.number_input("Short Strike", min_value=0.0, format="%.2f")
    with c2:
        expiration = st.date_input("Expiration Date")
        long_strike = st.number_input("Long Strike", min_value=0.0, format="%.2f")
    with c3:
        entry_date = st.date_input("Entry Date")
        credit = st.number_input("Credit Received", min_value=0.0, format="%.2f")

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
            st.success(f"Position initialized for {ticker}. Backend will sync data shortly.")

st.markdown("---")

# ---------------- Display Trades ----------------
st.subheader("Active Portfolio")

if not st.session_state.trades:
    st.info("No active trades.")
else:
    for i, t in enumerate(st.session_state.trades):
        cached = t.get("cached", {})

        current_dte = days_to_expiry(t["expiration"])
        entry_dte = get_entry_dte(t["entry_date"], t["expiration"])
        
        width = abs(t["short_strike"] - t["long_strike"])
        max_gain = t["credit"]
        max_loss = width - t["credit"]

        # Backend Data
        current_price = cached.get("current_price")
        
        # Support both old "abs_delta" and new "net_delta" structures
        abs_delta = cached.get("abs_delta") 
        if abs_delta is None and cached.get("delta"): 
             abs_delta = abs(cached.get("delta"))

        spread_value = cached.get("spread_value_percent")
        profit_pct = cached.get("current_profit_percent")
        rules = cached.get("rule_violations", {})

        # --- Status Logic ---
        status_msg = "Status Nominal"
        status_icon = "✅"
        status_color = "green"

        if rules.get("other_rules", False):
            status_icon = "⚠️"
            status_color = "#d32f2f" # Red
            if abs_delta and abs_delta >= 0.40:
                status_msg = "Short Delta High"
            elif spread_value and spread_value >= 150:
                status_msg = "Spread Value High"
            elif current_dte <= 7:
                status_msg = "Expiration Imminent"
        
        if profit_pct and profit_pct >= 50:
            status_icon = "💰" 
            status_msg = "Profit Target Reached"
            status_color = "green"

        # --- Color Coding ---
        delta_color = "#d32f2f" if abs_delta and abs_delta >= 0.40 else "green"
        delta_val = f"{abs_delta:.2f}" if abs_delta is not None else "Pending"

        spread_color = "#d32f2f" if spread_value and spread_value >= 150 else "green"
        spread_val = f"{spread_value:.0f}" if spread_value is not None else "Pending"

        dte_color = "#d32f2f" if current_dte <= 7 else "green"

        if profit_pct is None:
            profit_color = "inherit"
            profit_val = "Pending"
        else:
            profit_val = f"{profit_pct:.1f}"
            if profit_pct >= 50:
                profit_color = "green"
            elif profit_pct < 0:
                profit_color = "#d32f2f"
            else:
                profit_color = "#e6b800"

        cols = st.columns([3, 4])

        # -------- LEFT CARD (Details + Close Button) --------
        with cols[0]:
            # --- Ticker & Change Logic ---
            day_change = cached.get("day_change_percent", 0.0)
            
            if day_change is None: day_change = 0.0
            
            if day_change > 0:
                change_color = "green" 
                arrow = "▲"
                change_str = f"{day_change:.2f}%"
            elif day_change < 0:
                change_color = "#d32f2f"
                arrow = "▼"
                change_str = f"{abs(day_change):.2f}%" 
            else:
                change_color = "gray"
                arrow = ""
                change_str = "0.00%"

            st.markdown(f"""
            <div style="line-height: 1.4; font-size: 15px;">
                <h3 style="margin-bottom: 5px; display: flex; align-items: center; gap: 10px;">
                    {t['ticker']} 
                    <span style="color: {change_color}; font-size: 0.85em;">
                        {arrow} {change_str}
                    </span>
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2px;">
                    <div><strong>Short:</strong> {t['short_strike']}</div>
                    <div><strong>Max Gain:</strong> {format_money(max_gain)}</div>
                    <div><strong>Long:</strong> {t['long_strike']}</div>
                    <div><strong>Max Loss:</strong> {format_money(max_loss)}</div>
                    <div style="grid-column: span 2;"><strong>Exp:</strong> {t['expiration']}</div>
                    <div style="grid-column: span 2;"><strong>Width:</strong> {width:.2f}</div>
                </div>
                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; color: {status_color}; font-weight: bold;">
                   {status_icon} {status_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            # --- Close / Log Logic ---
            if st.button("Close Position / Log", key=f"btn_close_{i}"):
                st.session_state[f"close_mode_{i}"] = True

            # If close mode is active for this trade
            if st.session_state.get(f"close_mode_{i}", False):
                with st.container():
                    st.markdown("---")
                    st.info("📉 Closing Position & Logging to Journal")
                    
                    with st.form(key=f"close_form_{i}"):
                        col_log1, col_log2 = st.columns(2)
                        with col_log1:
                            # Auto-Calculate Price
                            default_debit = 0.0
                            current_short = t.get("cached", {}).get("short_option_price")
                            current_long = t.get("cached", {}).get("long_option_price")
                            
                            if current_short is not None and current_long is not None:
                                est_price = current_short - current_long
                                if est_price > 0:
                                    default_debit = est_price

                            debit_paid = st.number_input(
                                "Debit Paid ($)", 
                                min_value=0.0, 
                                value=float(f"{default_debit:.2f}"), 
                                step=0.01,
                                help="Enter the price you paid to buy back the spread (positive number)."
                            )
                        
                        with col_log2:
                            close_notes = st.text_area("Notes / Reason", height=70, placeholder="e.g. 50% profit target hit...")
                        
                        submit_close = st.form_submit_button("Confirm Close & Log")
                        
                        if submit_close:
                            success = False
                            
                            if drive_service:
                                success = log_trade_to_csv(drive_service, t, debit_paid, close_notes)
                                if success:
                                    st.success(f"Logged {t['ticker']} to 'trade_journal.csv'")
                                else:
                                    st.error("Could not write to Drive journal.")
                            else:
                                st.warning("Drive service not active. Removing without log.")
                                success = True 
                            
                            if success:
                                st.session_state.trades.pop(i)
                                if drive_service:
                                    save_to_drive(drive_service, st.session_state.trades)
                                del st.session_state[f"close_mode_{i}"]
                                st.experimental_rerun()

                    if st.button("Cancel", key=f"cancel_{i}"):
                        del st.session_state[f"close_mode_{i}"]
                        st.experimental_rerun()

        # -------- RIGHT CARD (Chart) --------
        with cols[1]:
            st.markdown(
                f"""
                <div style="font-size: 14px; margin-bottom: 10px;">
                    <div>Short-delta: <strong style='color:{delta_color}'>{delta_val}</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not exceed 0.40)</span></div>
                    <div>Spread Value: <strong style='color:{spread_color}'>{spread_val}%</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not exceed 150%)</span></div>
                    <div>DTE: <strong style='color:{dte_color}'>{current_dte}</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must not be less than 7)</span></div>
                    <div>Profit: <strong style='color:{profit_color}'>{profit_val}%</strong> <span style='color:gray; font-size:0.85em; display:block; margin-bottom:4px;'>(Must sell between 50-75%)</span></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- CHART LOGIC: Dark Candles Patch ---
            price_hist = t.get("cached", {}).get("price_history", [])
            crit_price = t.get("cached", {}).get("critical_price_040")
            
            if price_hist:
                try:
                    df_chart = pd.DataFrame(price_hist)
                    # Map JSON keys to DataFrame columns
                    df_chart['Date'] = pd.to_datetime(df_chart['date'])
                    df_chart['Close'] = df_chart['close']
                    # Ensure OHLC exists for candles
                    df_chart['Open'] = df_chart['open'] if 'open' in df_chart.columns else df_chart['close']
                    df_chart['High'] = df_chart['high'] if 'high' in df_chart.columns else df_chart['close']
                    df_chart['Low'] = df_chart['low'] if 'low' in df_chart.columns else df_chart['close']
                    
                    trade_start_ts = pd.Timestamp(t['entry_date'])
                    expiration_ts = pd.Timestamp(t['expiration'])
                    
                    fig = plot_spread_chart(
                        df=df_chart,
                        trade_start_date=trade_start_ts,
                        expiration_date=expiration_ts,
                        short_strike=t['short_strike'],
                        long_strike=t['long_strike'],
                        crit_price=crit_price
                    )
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Chart Error: {e}")
            else:
                st.caption("Initializing market data...")

        # Divider between trades
        st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

# ---------------- Manual Controls ----------------
st.write("### Data Sync")
ctl1, ctl2, ctl_spacer = st.columns([1.5, 1.5, 5])

with ctl1:
    if st.button("💾 Save all trades to Google Drive"):
        saved = False
        if drive_service:
            try:
                saved = save_to_drive(drive_service, st.session_state.trades)
            except Exception:
                saved = False
        if saved:
            st.success("Saved successfully.")
        else:
            st.error("Failed to save.")

with ctl2:
    if st.button("📥 Reload trades from Google Drive"):
        if drive_service:
            loaded = load_from_drive(drive_service)
            if loaded is not None:
                st.session_state.trades = loaded
                st.success("Loaded trades.")
                st.experimental_rerun()
            else:
                st.info("No trades found/failed.")

st.markdown("---")

# ---------------- External Tools Section ----------------
st.subheader("External Tools")
tools_c1, tools_c2, tools_c3, tools_c4 = st.columns(4)

with tools_c1:
    st.link_button("TradingView", "https://www.tradingview.com/", use_container_width=True)
with tools_c2:
    st.link_button("Wealthsimple", "https://my.wealthsimple.com/app/home", use_container_width=True)
with tools_c3:
    screener_url = "https://optionmoves.com/screener?ticker=SPY%2C+NVDA%2C+AAPL%2C+MSFT%2C+GOOG%2C+AMZN%2C+META%2C+BRK.B%2C+TSLA%2C+AVGO%2C+LLY%2C+JPM%2C+UNH%2C+V%2C+MA%2C+JNJ%2C+XOM%2C+CVX%2C+PG%2C+PEP%2C+KO%2C+WMT%2C+BAC%2C+PFE%2C+NFLX%2C+ORCL%2C+ADBE%2C+INTC%2C+COST%2C+ABT%2C+VZ&strategy=put-credit-spread&expiryType=dte&dte=30&deltaStrikeType=delta&delta=0.30&spreadWidth=5"
    st.link_button("Option Screener", screener_url, use_container_width=True)
with tools_c4:
    st.link_button("IV Rank Check", "https://marketchameleon.com/", use_container_width=True)
