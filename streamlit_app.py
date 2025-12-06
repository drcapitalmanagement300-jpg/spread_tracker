# app.py (patched to use persistence.py + IV30 percent rank)
import streamlit as st
from datetime import date, datetime, timedelta
import json
import io
from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt

# Persistence (Google OAuth + Drive) — provided in persistence.py
from persistence import (
    ensure_logged_in,
    build_drive_service_from_session,
    save_to_drive,
    load_from_drive,
    logout,
)

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ----------------------------- IV30 Percent Rank helpers -----------------------------
# Color settings (use same basic green/red as your cards; brighter variants for "hyper" states)
ELEVATED_COLOR = "green"          # same as card
HYPER_ELEVATED_COLOR = "#66ff66"  # brighter green
LOW_COLOR = "red"                 # same as card
HYPER_LOW_COLOR = "#ff6666"       # brighter red

@st.cache_data(ttl=60 * 30)
def get_price_simple(ticker: str) -> Optional[float]:
    """Lightweight price fetch (used by IV computation)."""
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        return float(fi["last_price"])
    except Exception:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if hist is not None and not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return None

@st.cache_data(ttl=60 * 30)
def get_iv30_via_options(ticker: str) -> Optional[float]:
    """
    Attempt to compute a 30-day implied volatility by:
    - Finding the option expiration date closest to 30 days out
    - Averaging impliedVolatility of ATM-ish calls & puts (within ~2% of spot)
    Returns IV as a percentage (e.g., 25.3 for 25.3%).
    """
    try:
        tk = yf.Ticker(ticker)
        opts = tk.options
        if not opts:
            return None
        spot = get_price_simple(ticker)
        if spot is None:
            return None

        # find expiration closest to 30 DTE
        best_exp = None
        best_diff = None
        today = date.today()
        for exp_str in opts:
            try:
                exp_date = datetime.fromisoformat(exp_str).date()
            except Exception:
                # some expirations may be in different format; try parsing
                try:
                    exp_date = pd.to_datetime(exp_str).date()
                except Exception:
                    continue
            dte = (exp_date - today).days
            diff = abs(dte - 30)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_exp = exp_str

        if best_exp is None:
            return None

        calls, puts = None, None
        try:
            opt_chain = tk.option_chain(best_exp)
            calls, puts = opt_chain.calls, opt_chain.puts
        except Exception:
            return None

        if (calls is None or calls.empty) and (puts is None or puts.empty):
            return None

        # Choose ATM strikes within ~2% of spot
        low_strike = spot * 0.98
        high_strike = spot * 1.02

        ivs = []
        if calls is not None and not calls.empty:
            atm_calls = calls[(calls["strike"] >= low_strike) & (calls["strike"] <= high_strike)]
            if atm_calls is None or atm_calls.empty:
                # if none near ATM, sample median few strikes around median
                atm_calls = calls.iloc[max(0, len(calls)//2 - 2): min(len(calls), len(calls)//2 + 3)]
            if 'impliedVolatility' in atm_calls.columns:
                ivs += [float(x) * 100 for x in atm_calls['impliedVolatility'].dropna().values]

        if puts is not None and not puts.empty:
            atm_puts = puts[(puts["strike"] >= low_strike) & (puts["strike"] <= high_strike)]
            if atm_puts is None or atm_puts.empty:
                atm_puts = puts.iloc[max(0, len(puts)//2 - 2): min(len(puts), len(puts)//2 + 3)]
            if 'impliedVolatility' in atm_puts.columns:
                ivs += [float(x) * 100 for x in atm_puts['impliedVolatility'].dropna().values]

        if not ivs:
            return None

        # Return mean IV
        return float(np.mean(ivs))
    except Exception:
        return None

@st.cache_data(ttl=60 * 60 * 6)
def get_iv30_percent_rank(ticker: str) -> Optional[float]:
    """
    Compute IV30 Percent Rank (0-100).
    1) Try to get today's IV30 from options (preferred).
    2) If not available, fall back to a realized 30-day rolling vol proxy.
    3) Build a 1-year historical series of the same metric and compute percentile rank.
    """
    try:
        # preferred: option-based IV30
        current_iv30 = get_iv30_via_options(ticker)

        # build historical series (1 year) using realized vol proxy when option IVs aren't available historically
        tk = yf.Ticker(ticker)
        prices = tk.history(period="1y", interval="1d")["Close"].dropna()
        if prices.empty:
            # if no price history, fallback to option-based current iv only
            if current_iv30 is not None:
                # without history we can't compute rank reliably; return None
                return None
            return None

        # realized vol proxy: 30-day rolling std (annualized)
        returns = np.log(prices / prices.shift(1)).dropna()
        rolling30 = returns.rolling(window=30).std() * np.sqrt(252)  # annualized
        # Convert to percent
        rolling30_pct = (rolling30 * 100).dropna()

        # If we don't have a option-based current_iv30, approximate it with the latest realized 30-day vol
        if current_iv30 is None:
            if not rolling30_pct.empty:
                current_iv30 = float(rolling30_pct.iloc[-1])
            else:
                return None

        # Percent rank: percentage of historical values below current
        if rolling30_pct.empty:
            return None

        rank = float((rolling30_pct < current_iv30).sum() / len(rolling30_pct) * 100)
        return round(rank, 1)
    except Exception:
        return None

def iv_rank_status(rank: Optional[float]) -> str:
    if rank is None:
        return "N/A"
    if rank >= 70:
        return "Hyper Elevated"
    if rank >= 50:
        return "Elevated"
    if rank >= 30:
        return "Low"
    return "Hyper Low"

def iv_rank_color(rank: Optional[float]) -> str:
    """Return color string/hex for the rank per your specification."""
    if rank is None:
        return "black"
    if rank >= 70:
        return HYPER_ELEVATED_COLOR
    if rank >= 50:
        return ELEVATED_COLOR
    if rank >= 30:
        return LOW_COLOR
    return HYPER_LOW_COLOR

# ----------------------------- App core (UI & logic) -----------------------------

# Ensure the user is logged-in via OAuth (this will show sign-in UI and stop the app
# if the user is not authenticated). We catch exceptions so the app can still run
# locally if you decide to not sign in.
try:
    ensure_logged_in()
except Exception:
    # If ensure_logged_in raised (for example, missing secrets), allow the app to continue
    # but warn the user.
    st.warning("Google OAuth not available. You can still use the app locally but Drive persistence will be disabled.")
    # Do not stop; build_drive_service_from_session will likely return None below.

# Build Drive service (may be None if not signed in)
drive_service = None
try:
    drive_service = build_drive_service_from_session()
except Exception:
    drive_service = None

# small logout button in header area
_, logout_col = st.columns([9, 1])
with logout_col:
    if st.button("Log out"):
        try:
            logout()
        except Exception:
            # If logout fails for any reason, still clear credentials
            st.session_state.pop("credentials", None)
            st.success("Logged out (local). Reload the page to sign in again.")
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ------------------- Helpers (same as your original app) -------------------
def init_state():
    if "trades" not in st.session_state:
        loaded = []
        if drive_service:
            try:
                loaded = load_from_drive(drive_service) or []
            except Exception:
                loaded = []
        st.session_state.trades = loaded if loaded else []

def days_to_expiry(expiry_date: date) -> int:
    return max((expiry_date - date.today()).days, 0)

def compute_derived(trade: dict) -> dict:
    short = float(trade["short_strike"])
    long = float(trade["long_strike"])
    credit = float(trade.get("credit", 0) or 0)
    width = abs(long - short)
    max_gain = credit
    max_loss = max(width - credit, 0)
    breakeven = short + credit
    dte = days_to_expiry(trade["expiration"])
    return {
        "width": width,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakeven": breakeven,
        "dte": dte
    }

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

@st.cache_data(ttl=60)
def get_price(ticker: str):
    try:
        data = yf.Ticker(ticker).fast_info
        return float(data["last_price"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiration: str):
    try:
        ticker_obj = yf.Ticker(ticker)
        opt_chain = ticker_obj.option_chain(expiration)
        return opt_chain.calls, opt_chain.puts
    except Exception:
        return None, None

def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        return norm.cdf(d1)
    elif option_type.lower() == 'put':
        return norm.cdf(d1) - 1
    else:
        return None

def get_leg_data(ticker: str, expiration: date, strike: float, option_type='put'):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None, None
    leg_row = puts[puts['strike'] == strike]
    if leg_row.empty:
        return None, None
    price = leg_row['lastPrice'].values[0] if 'lastPrice' in leg_row.columns else None
    iv = leg_row['impliedVolatility'].values[0] * 100 if 'impliedVolatility' in leg_row.columns else None
    return price, iv

def get_short_leg_data(trade: dict):
    short_price, iv = get_leg_data(trade["ticker"], trade["expiration"], float(trade["short_strike"]), 'put')
    current_price = get_price(trade['ticker'])
    delta = None
    if current_price and iv:
        T = days_to_expiry(trade["expiration"]) / 365
        sigma = iv / 100
        r = 0.05
        delta = bsm_delta('put', current_price, float(trade["short_strike"]), T, r, sigma)
    return delta, iv, short_price

def get_long_leg_data(trade: dict):
    long_price, _ = get_leg_data(trade["ticker"], trade["expiration"], float(trade["long_strike"]), 'put')
    return long_price

def compute_spread_value(short_option_price, long_option_price, width, credit):
    if short_option_price is None or long_option_price is None or width - credit <= 0:
        return None
    spread_mark = short_option_price - long_option_price
    max_loss = width - credit
    return (spread_mark / max_loss) * 100

def compute_current_profit(short_price, long_price, credit, width):
    if short_price is None or long_price is None or credit <= 0:
        return None
    spread_value = short_price - long_price
    current_profit = credit - spread_value
    return max(0, min((current_profit / credit) * 100, 100))

def fetch_short_iv(ticker, short_strike, expiration):
    _, puts = get_option_chain(ticker, expiration.isoformat())
    if puts is None or puts.empty:
        return None
    short_row = puts[puts['strike'] == short_strike]
    if short_row.empty or 'impliedVolatility' not in short_row.columns:
        return None
    iv = short_row['impliedVolatility'].values[0] * 100
    return iv

def evaluate_rules(trade, derived, current_price, delta, current_iv, short_option_price, long_option_price):
    rule_violations = {"other_rules": False, "iv_rule": False}
    abs_delta = abs(delta) if delta is not None else None
    if abs_delta is not None and abs_delta >= 0.40:
        rule_violations["other_rules"] = True
    spread_value_percent = compute_spread_value(short_option_price, long_option_price, derived["width"], trade["credit"])
    if spread_value_percent is not None and spread_value_percent >= 150:
        rule_violations["other_rules"] = True
    if derived["dte"] <= 7:
        rule_violations["other_rules"] = True
    entry_iv = trade.get("entry_iv")
    if entry_iv and current_iv and current_iv > entry_iv:
        rule_violations["iv_rule"] = True
    return rule_violations, abs_delta, spread_value_percent

# ----------------------------- Initialize session & UI -----------------------------
init_state()

with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        ticker = st.text_input("Ticker (e.g. AAPL)").upper()
        # show IV30 Rank under the ticker with color coding
        iv30_rank = None
        iv30_status = None
        iv30_color = "black"
        if ticker:
            try:
                iv30_rank = get_iv30_percent_rank(ticker)
                iv30_status = iv_rank_status(iv30_rank)
                iv30_color = iv_rank_color(iv30_rank)
            except Exception:
                iv30_rank = None
                iv30_status = "N/A"
                iv30_color = "black"

            # Color-coded IV30 percent rank display
            if iv30_rank is None:
                st.markdown("<div>IV30 Percent Rank: <strong>N/A</strong> — <em>N/A</em></div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    html = f"<div style='color:{iv30_color}'>IV30 Percent Rank: <strong>{iv30_rank:.1f}%</strong></div>",
                    unsafe_allow_html=True
                )

        short_strike = st.number_input("Short strike", min_value=0.0, format="%.2f")
        long_strike = st.number_input("Long strike", min_value=0.0, format="%.2f")
    with col2:
        expiration = st.date_input("Expiration date", value=date.today())
        credit = st.number_input("Credit received (per share)", min_value=0.0, format="%.2f")
    with col3:
        entry_date = st.date_input("Entry date", value=date.today())
        notes = st.text_input("Notes (optional)")
        st.write("")
    submitted = st.form_submit_button("Add trade for monitoring")
    if submitted:
        if not ticker:
            st.warning("Please provide a ticker symbol.")
        elif long_strike >= short_strike:
            st.warning("For a put credit spread, long strike should be LOWER than short strike.")
        else:
            auto_iv = fetch_short_iv(ticker, short_strike, expiration)
            trade = {
                "id": f"{ticker}-{short_strike}-{long_strike}-{expiration.isoformat()}",
                "ticker": ticker,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration,
                "credit": credit,
                "entry_date": entry_date,
                "entry_iv": auto_iv,
                # store entry IV30 rank
                "entry_iv30_rank": iv30_rank,
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)

            # Try to save to Drive; if Drive not configured, show success locally
            saved_to_drive = False
            if drive_service:
                try:
                    saved_to_drive = save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    saved_to_drive = False

            if saved_to_drive:
                st.success(f"Added {ticker} — saved to Drive. Entry IV: {auto_iv if auto_iv else 'N/A'}")
            else:
                st.success(f"Added {ticker} locally. (Drive not configured or save failed)")

st.markdown("---")

st.subheader("Active Trades")
if not st.session_state.trades:
    st.info("No trades added yet. Use the form above to add your first spread.")
else:
    for i, t in enumerate(st.session_state.trades):
        derived = compute_derived(t)
        current_price = get_price(t['ticker'])
        delta, current_iv, short_option_price = get_short_leg_data(t)
        long_option_price = get_long_leg_data(t)
        current_profit_percent = compute_current_profit(short_option_price, long_option_price, t["credit"], derived["width"])
        rule_violations, abs_delta, spread_value_percent = evaluate_rules(
            t, derived, current_price, delta, current_iv, short_option_price, long_option_price
        )

        abs_delta_str = f"{abs_delta:.2f}" if abs_delta is not None else "-"
        spread_value_str = f"{spread_value_percent:.0f}%" if spread_value_percent is not None else "-"
        current_profit_str = f"{current_profit_percent:.1f}%" if current_profit_percent is not None else "-"
        current_price_str = f"{current_price:.2f}" if current_price is not None else "-"

        if rule_violations["other_rules"]:
            status_icon = "❌"
            status_text = "Some critical rules are violated."
        elif rule_violations["iv_rule"]:
            status_icon = "⚠️"
            status_text = "Current IV exceeds entry IV."
        else:
            status_icon = "✅"
            status_text = "All rules are satisfied."

        card_cols = st.columns([3,3])
        with card_cols[0]:
            st.markdown(
                f"""
<div style='background-color:rgba(0,100,0,0.1); padding:15px; border-radius:10px; height:100%'>
Ticker: {t['ticker']}  <br>
Underlying Price: {current_price_str}  <br>
Short Strike: {t['short_strike']}  <br>
Long Strike: {t['long_strike']}  <br>
Spread Width: {derived['width']}  <br>
Expiration Date: {t['expiration']}  <br>
Current DTE: {derived['dte']}  <br>
Max Gain: {format_money(derived['max_gain'])}  <br>
Max Loss: {format_money(derived['max_loss'])}  
</div>
""", unsafe_allow_html=True)

            st.markdown(f"<div style='margin-top:10px; font-size:20px'>{status_icon} {status_text}</div>", unsafe_allow_html=True)

        with card_cols[1]:
            delta_color = "red" if abs_delta is not None and abs_delta >= 0.40 else "green"
            spread_color = "red" if spread_value_percent is not None and spread_value_percent >= 150 else "green"
            dte_color = "red" if derived['dte'] <= 7 else "green"

            if current_profit_percent is None:
                profit_color = "black"
            elif current_profit_percent < 50:
                profit_color = "green"
            elif 50 <= current_profit_percent <= 75:
                profit_color = "yellow"
            else:
                profit_color = "red"

            if current_iv is None or t.get("entry_iv") is None:
                iv_color = "black"
            elif current_iv == t.get("entry_iv"):
                iv_color = "yellow"
            elif current_iv > t.get("entry_iv"):
                iv_color = "red"
            else:
                iv_color = "green"

            # Entry/Current IV30 Percent Rank display (no special color here per your request)
            entry_iv30_rank = t.get("entry_iv30_rank", None)
            try:
                current_iv30_rank = get_iv30_percent_rank(t["ticker"])
            except Exception:
                current_iv30_rank = None

            entry_rank_display = f\"{entry_iv30_rank:.1f}%\" if isinstance(entry_iv30_rank, (int, float)) else (str(entry_iv30_rank) or "N/A")
            current_rank_display = f\"{current_iv30_rank:.1f}%\" if isinstance(current_iv30_rank, (int, float)) else (str(current_iv30_rank) or "N/A")

            # Avoid formatting errors if entry_iv/current_iv is None
            entry_iv_display = f"{t.get('entry_iv'):.1f}%" if isinstance(t.get("entry_iv"), (int, float)) else (str(t.get("entry_iv")) or "N/A")
            current_iv_display = f"{current_iv:.1f}%" if isinstance(current_iv, (int, float)) else (str(current_iv) or "N/A")

            st.markdown(
                f"""
Short Delta: <span style='color:{delta_color}'>{abs_delta_str}</span> | Must be less than or equal to 0.40 <br>
Spread Value: <span style='color:{spread_color}'>{spread_value_str}</span> | Must be less than or equal to 150% of credit <br>
DTE: <span style='color:{dte_color}'>{derived['dte']}</span> | Must be greater than 7 <br>
Current Profit: <span style='color:{profit_color}'>{current_profit_str}</span> | 50-75% Max profit target <br>
Entry IV30 Percent Rank: {entry_rank_display} | Current IV30 Percent Rank: <span style='font-weight:600'>{current_rank_display}</span><br>
Entry IV: {entry_iv_display} | Current IV: <span style='color:{iv_color}'>{current_iv_display}</span>
""", unsafe_allow_html=True)

            # PnL chart
            dte_range = list(range(derived["dte"] + 1))
            profit_values = [current_profit_percent if current_profit_percent is not None else 0]*len(dte_range)
            pnl_df = pd.DataFrame({"DTE": dte_range, "Profit %": profit_values})

            base_chart = alt.Chart(pnl_df).mark_line(point=True).encode(
                x=alt.X('DTE', title='Days to Expiration', scale=alt.Scale(domain=(derived["dte"], 0))),
                y=alt.Y('Profit %', title='Current Profit %', scale=alt.Scale(domain=(0,100), nice=False),
                        axis=alt.Axis(tickMinStep=10, tickCount=11))
            ).properties(height=250)

            line_50 = alt.Chart(pd.DataFrame({'y':[50]})).mark_rule(strokeDash=[5,5]).encode(y='y')
            line_75 = alt.Chart(pd.DataFrame({'y':[75]})).mark_rule(strokeDash=[5,5]).encode(y='y')
            vline = alt.Chart(pd.DataFrame({'DTE':[derived['dte']]})).mark_rule(strokeDash=[5,5]).encode(x='DTE')

            final_chart = base_chart + line_50 + line_75 + vline
            st.altair_chart(final_chart, use_container_width=True)

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            # Try saving updated trades to Drive
            saved = False
            if drive_service:
                try:
                    saved = save_to_drive(drive_service, st.session_state.trades)
                except Exception:
                    saved = False

            if saved:
                st.success("Saved updated trades to Drive.")
            else:
                st.warning("Removed locally but failed to save to Drive (or Drive not configured).")
            try:
                st.experimental_rerun()
            except Exception:
                pass

st.markdown("---")

# Manual Save/Load (uses persistence API)
colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save all trades to Google Drive now"):
        saved = False
        if drive_service:
            try:
                saved = save_to_drive(drive_service, st.session_state.trades)
            except Exception:
                saved = False
        if saved:
            st.success("Saved to Drive successfully.")
        else:
            st.error("Failed to save to Drive. Check logs or ensure you're signed in.")

with colB:
    if st.button("📥 Reload trades from Google Drive"):
        loaded = []
        if drive_service:
            try:
                loaded = load_from_drive(drive_service) or []
            except Exception:
                loaded = []
        if loaded:
            st.session_state.trades = loaded
            st.success("Loaded trades from Drive.")
            try:
                st.experimental_rerun()
            except Exception:
                pass
        else:
            st.info("No trades found on Drive (or load failed).")

st.caption("Spread value uses actual option prices — alerts accurate, delta BSM-based, entry IV auto-captured.")
