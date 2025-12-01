# app.py
import streamlit as st
from datetime import date, datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import json
import io
from typing import List, Dict, Any, Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ----------------------------- Google Drive configuration -----------------------------
# Put optional DRIVE_FOLDER_ID in Streamlit secrets if you want files stored inside a specific folder.
# In Streamlit secrets:
# [gcp_service_account]
# ... your service account JSON fields ...
# DRIVE_FOLDER_ID = "your_folder_id_here"   # optional
DRIVE_FILE_NAME = "credit_spreads.json"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

# ----------------------------- Drive helpers -----------------------------
@st.cache_resource
def init_drive() -> Optional[object]:
    """
    Initialize Google Drive service using service account stored in st.secrets["gcp_service_account"].
    Returns the drive service object or None.
    """
    try:
        if "gcp_service_account" not in st.secrets:
            st.warning("No gcp_service_account found in Streamlit secrets. Drive persistence disabled.")
            return None

        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=DRIVE_SCOPES)
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        # optional folder id in secrets
        folder_id = st.secrets.get("DRIVE_FOLDER_ID") if hasattr(st, "secrets") else None
        if folder_id:
            st.info(f"Using Drive folder ID: {folder_id}")
        # show service account email (safe debug)
        sa_email = creds_dict.get("client_email")
        if sa_email:
            st.write(f"Drive service account: {sa_email}")
        return service
    except Exception as e:
        st.error(f"Google Drive init failed: {e}")
        return None

def _get_folder_id() -> Optional[str]:
    """Return DRIVE_FOLDER_ID from secrets if set."""
    try:
        return st.secrets.get("DRIVE_FOLDER_ID")  # may be None
    except Exception:
        return None

def _find_file_id(service, filename: str) -> Optional[str]:
    """Find file id by filename, optionally scoped to folder if DRIVE_FOLDER_ID set."""
    if service is None:
        return None
    try:
        folder_id = _get_folder_id()
        if folder_id:
            q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
        else:
            q = f"name = '{filename}' and trashed = false"
        resp = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
        files = resp.get("files", [])
        if not files:
            return None
        return files[0]["id"]
    except Exception as e:
        st.error(f"Drive find file error: {e}")
        return None

def _download_file(service, file_id: str) -> Optional[str]:
    """Download file content and return as string, or None on error."""
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read().decode("utf-8")
    except Exception as e:
        st.error(f"Drive download error: {e}")
        return None

def _upload_file(service, filename: str, content_str: str) -> bool:
    """Upload or update a file in Drive. Returns True on success."""
    try:
        folder_id = _get_folder_id()
        file_id = _find_file_id(service, filename)
        fh = io.BytesIO(content_str.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)

        if file_id:
            # Update existing file
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            # Create new file; optionally place in specific folder
            body = {"name": filename}
            if folder_id:
                body["parents"] = [folder_id]
            service.files().create(body=body, media_body=media).execute()
        return True
    except Exception as e:
        st.error(f"Drive upload error: {e}")
        return False

def save_to_drive(service, trades: List[Dict[str, Any]]) -> bool:
    """Serialize trades and save to Google Drive. Returns True on success."""
    if service is None:
        st.warning("Drive service not available; cannot save.")
        return False
    # convert date objects to ISO strings
    serializable = []
    for t in trades:
        ct = {}
        for k, v in t.items():
            if isinstance(v, (date, datetime)):
                ct[k] = v.isoformat()
            else:
                ct[k] = v
        serializable.append(ct)
    return _upload_file(service, DRIVE_FILE_NAME, json.dumps(serializable, indent=2))

def load_from_drive(service) -> List[Dict[str, Any]]:
    """Load trades JSON from Drive and convert ISO dates back to date objects."""
    if service is None:
        return []
    try:
        file_id = _find_file_id(service, DRIVE_FILE_NAME)
        if not file_id:
            return []
        raw = _download_file(service, file_id)
        if not raw:
            return []
        loaded = json.loads(raw)
        out = []
        for t in loaded:
            nt = {}
            for k, v in t.items():
                if isinstance(v, str) and k in ("expiration", "entry_date", "created_at"):
                    # try parse as date or datetime
                    try:
                        parsed = datetime.fromisoformat(v)
                        if k == "created_at":
                            nt[k] = v  # keep created_at as string (optional)
                        else:
                            nt[k] = parsed.date()
                    except Exception:
                        nt[k] = v
                else:
                    nt[k] = v
            out.append(nt)
        return out
    except Exception as e:
        st.error(f"Drive load error: {e}")
        return []

# Initialize drive service (cached)
drive = init_drive()

# ----------------------------- Original helpers (unchanged, cleaned) -----------------------------
def init_state():
    if "trades" not in st.session_state:
        # attempt to load from drive; if fail, start empty
        if drive:
            try:
                st.session_state.trades = load_from_drive(drive) or []
            except Exception:
                st.session_state.trades = []
        else:
            st.session_state.trades = []

def days_to_expiry(exp: date) -> int:
    return max((exp - date.today()).days, 0)

def compute_derived(t: Dict[str, Any]) -> Dict[str, Any]:
    short = float(t["short_strike"])
    long = float(t["long_strike"])
    credit = float(t.get("credit", 0) or 0)
    width = abs(long - short)
    max_gain = credit
    max_loss = max(width - credit, 0)
    return {
        "width": width,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "breakeven": short + credit,
        "dte": days_to_expiry(t["expiration"])
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

# ----------------------------- Initialize session -----------------------------
init_state()

# ----------------------------- Trade input form -----------------------------
with st.form("add_trade", clear_on_submit=True):
    st.subheader("Add new put credit spread")
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        ticker = st.text_input("Ticker (e.g. AAPL)").upper()
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
                "notes": notes,
                "created_at": datetime.utcnow().isoformat()
            }
            st.session_state.trades.append(trade)
            # save to drive immediately if available
            if drive:
                ok = save_to_drive(drive, st.session_state.trades)
                if ok:
                    st.success(f"Added {ticker} — saved to Drive. Entry IV: {auto_iv if auto_iv else 'N/A'}")
                else:
                    st.warning(f"Added {ticker} locally, but failed to save to Drive.")
            else:
                st.success(f"Added {ticker} locally. (Drive not configured)")

st.markdown("---")

# ----------------------------- Active Trades dashboard -----------------------------
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

            if current_iv is None or t["entry_iv"] is None:
                iv_color = "black"
            elif current_iv == t["entry_iv"]:
                iv_color = "yellow"
            elif current_iv > t["entry_iv"]:
                iv_color = "red"
            else:
                iv_color = "green"

            st.markdown(
                f"""
Short Delta: <span style='color:{delta_color}'>{abs_delta_str}</span> | Must be less than or equal to 0.40 <br>
Spread Value: <span style='color:{spread_color}'>{spread_value_str}</span> | Must be less than or equal to 150% of credit <br>
DTE: <span style='color:{dte_color}'>{derived['dte']}</span> | Must be greater than 7 <br>
Current Profit: <span style='color:{profit_color}'>{current_profit_str}</span> | 50-75% Max profit target <br>
Entry IV: {t['entry_iv']:.1f}% | Current IV: <span style='color:{iv_color}'>{current_iv:.1f}%</span>
""", unsafe_allow_html=True)

            # ------------------- PnL chart -------------------
            dte_range = list(range(derived["dte"] + 1))
            profit_values = [current_profit_percent if current_profit_percent is not None else 0]*len(dte_range)
            pnl_df = pd.DataFrame({
                "DTE": dte_range,
                "Profit %": profit_values
            })

            base_chart = alt.Chart(pnl_df).mark_line(point=True).encode(
                x=alt.X('DTE', title='Days to Expiration', scale=alt.Scale(domain=(derived["dte"], 0))),
                y=alt.Y('Profit %', title='Current Profit %', scale=alt.Scale(domain=(0,100), nice=False),
                        axis=alt.Axis(tickMinStep=10, tickCount=11))
            ).properties(height=250)

            line_50 = alt.Chart(pd.DataFrame({'y':[50]})).mark_rule(color='yellow', strokeDash=[5,5]).encode(y='y')
            line_75 = alt.Chart(pd.DataFrame({'y':[75]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y')
            vline = alt.Chart(pd.DataFrame({'DTE':[derived['dte']]})).mark_rule(color='blue', strokeDash=[5,5]).encode(x='DTE')

            final_chart = base_chart + line_50 + line_75 + vline
            st.altair_chart(final_chart, use_container_width=True)

        if st.button("Remove", key=f"remove_{i}"):
            st.session_state.trades.pop(i)
            if drive:
                ok = save_to_drive(drive, st.session_state.trades)
                if ok:
                    st.success("Saved updated trades to Drive.")
                else:
                    st.warning("Removed locally but failed to save to Drive.")
            st.experimental_rerun()

st.markdown("---")
st.caption("Spread value uses actual option prices — alerts accurate, delta BSM-based, entry IV auto-captured.")
