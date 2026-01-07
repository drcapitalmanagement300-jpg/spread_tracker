import json
import io
import os
import time
import logging
import requests
from datetime import datetime, date, timedelta, timezone

import numpy as np
from scipy.stats import norm
import yfinance as yf

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

try:
    from google_drive import DriveLockManager
except ImportError:
    logging.warning("Could not import DriveLockManager. Running without locks.")
    DriveLockManager = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]

FILE_ID = os.environ.get("GDRIVE_FILE_ID")
CREDS_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

class MarketDataManager:
    def __init__(self):
        self._price_cache = {}    
        self._chain_cache = {}    
        self._ticker_objs = {}
        self._market_regime_cache = None    

    def _get_yf_ticker(self, ticker):
        if ticker not in self._ticker_objs:
            self._ticker_objs[ticker] = yf.Ticker(ticker)
        return self._ticker_objs[ticker]

    def get_market_regime(self):
        if self._market_regime_cache:
            return self._market_regime_cache
        try:
            spy = self._get_yf_ticker("SPY")
            hist = spy.history(period="1y")
            if hist.empty or len(hist) < 200:
                return True, 0.0, 0.0
            current_price = hist['Close'].iloc[-1]
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            is_safe = current_price > sma_200
            self._market_regime_cache = (is_safe, current_price, sma_200)
            return is_safe, current_price, sma_200
        except Exception as e:
            logger.error(f"Failed to fetch Market Regime: {e}")
            return True, 0.0, 0.0

    def get_price_data(self, ticker):
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        ticker_obj = self._get_yf_ticker(ticker)
        price = None
        change_pct = 0.0
        history_list = []
        for attempt in range(3):
            try:
                hist = ticker_obj.history(period="1mo")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    if len(hist) >= 2:
                        prev_close = hist['Close'].iloc[-2]
                        if prev_close > 0:
                            change_pct = ((price - prev_close) / prev_close) * 100
                    hist_reset = hist.reset_index()
                    history_list = []
                    for _, row in hist_reset.iterrows():
                        d_str = row['Date'].strftime('%Y-%m-%d')
                        history_list.append({
                            "date": d_str,
                            "open": round(row['Open'], 2),
                            "high": round(row['High'], 2),
                            "low": round(row['Low'], 2),
                            "close": round(row['Close'], 2)
                        })
                    break
            except Exception as e:
                logger.warning(f"Data fetch failed for {ticker}: {e}")
                time.sleep(2)
        if price:
            self._price_cache[ticker] = (price, change_pct, history_list)
            return price, change_pct, history_list
        return None, None, []

    def get_chain(self, ticker, expiration):
        cache_key = (ticker, expiration)
        if cache_key in self._chain_cache:
            return self._chain_cache[cache_key]
        ticker_obj = self._get_yf_ticker(ticker)
        chain = None
        try:
            if expiration not in ticker_obj.options:
                return None
        except: pass 
        for attempt in range(3):
            try:
                chain = ticker_obj.option_chain(expiration)
                if chain is not None: break
            except Exception as e:
                logger.warning(f"Chain fetch failed for {ticker}: {e}")
                time.sleep(2)
        if chain:
            self._chain_cache[cache_key] = chain
            return chain
        return None

def send_discord_alert(ticker, description, color=15158332):
    if not DISCORD_WEBHOOK_URL: return
    payload = {
        "username": "DR Capital Monitor",
        "embeds": [{
            "title": f"🔔 {ticker}",
            "description": description,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "DR Capital Portfolio System"}
        }]
    }
    try: requests.post(DISCORD_WEBHOOK_URL, json=payload)
    except: pass

def handle_heartbeat(updated_trades):
    now_utc = datetime.now(timezone.utc)
    if now_utc.hour in [13, 14]:
        if not updated_trades: return
        last_hb = updated_trades[0].get("last_heartbeat_date")
        today_str = date.today().isoformat()
        if last_hb != today_str:
            total = len(updated_trades)
            breaches = sum(1 for t in updated_trades if t.get("cached", {}).get("rule_violations", {}).get("other_rules"))
            summary = f"☀️ **Morning Check**\n✅ **Status:** Online\n📊 **Positions:** {total}\n🚨 **Breaches:** {breaches}"
            send_discord_alert("System Heartbeat", summary, color=3447003) 
            updated_trades[0]["last_heartbeat_date"] = today_str

def get_drive_service():
    if not CREDS_JSON or not FILE_ID:
        raise ValueError("Missing Env Vars")
    creds = Credentials.from_service_account_info(json.loads(CREDS_JSON), scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

def download_json(service):
    fh = io.BytesIO()
    request = service.files().get_media(fileId=FILE_ID)
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done: _, done = downloader.next_chunk()
    fh.seek(0)
    try: return json.load(fh)
    except: return []

def upload_json(service, data):
    fh = io.BytesIO(json.dumps(data, indent=2).encode("utf-8"))
    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=False)
    service.files().update(fileId=FILE_ID, media_body=media).execute()

def days_to_expiry(expiry_str):
    if not expiry_str: return 0
    try: expiry = date.fromisoformat(expiry_str)
    except: return 0
    return max((expiry - date.today()).days, 0)

# --- MATH FUNCTIONS ---

def bs_price(option_type, S, K, T, r, sigma):
    """Calculates Black-Scholes price."""
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_stop_loss_price(target_value, option_type, short_K, long_K, T, r, short_sigma, long_sigma):
    """
    Finds the underlying price S where the spread value (Short - Long) equals target_value.
    Uses a binary search with fixed Sigma to avoid skewed data artifacts.
    """
    width = abs(short_K - long_K)
    if target_value >= width: return None 
    if target_value <= 0: return None

    # Search range: from effectively 0 to double the short strike
    low = 0.01
    high = short_K * 2.0
    
    # Binary Search
    for _ in range(20):
        mid = (low + high) / 2
        p_short = bs_price(option_type, mid, short_K, T, r, short_sigma)
        p_long = bs_price(option_type, mid, long_K, T, r, long_sigma)
        
        spread_val = p_short - p_long
        
        # Puts: Lower Price = Higher Value.
        # If Current Val > Target -> We are too deep. Need Higher Price.
        if option_type == "put":
            if spread_val > target_value:
                low = mid
            else:
                high = mid
        else:
            if spread_val > target_value:
                high = mid
            else:
                low = mid
            
    return high

def calculate_greeks(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    delta = cdf_d1 if option_type == "call" else cdf_d1 - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        theta_annual = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta_annual = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "theta": theta_annual / 365.0}

def get_option_data(data_manager, ticker, expiration_str, short_strike, long_strike):
    try:
        chain = data_manager.get_chain(ticker, expiration_str)
        if chain is None: return None, None, None, None
        puts = chain.puts
        short_row = puts[puts['strike'] == short_strike]
        long_row = puts[puts['strike'] == long_strike]
        if short_row.empty or long_row.empty: return None, None, None, None
        return (float(short_row['lastPrice'].values[0]), 
                float(long_row['lastPrice'].values[0]), 
                float(short_row['impliedVolatility'].values[0]), 
                float(long_row['impliedVolatility'].values[0]))
    except: return None, None, None, None

def update_trade(trade, data_manager):
    ticker = trade.get("ticker")
    expiry_str = trade.get("expiration")
    try:
        short_strike = float(trade.get("short_strike", 0))
        long_strike = float(trade.get("long_strike", 0))
        credit_received = float(trade.get("credit", 0))
        contracts = int(trade.get("contracts", 1)) 
    except ValueError: return trade 

    current_price, day_change_pct, price_history = data_manager.get_price_data(ticker)
    short_price, long_price, short_iv, long_iv = get_option_data(data_manager, ticker, expiry_str, short_strike, long_strike)
    is_market_safe, _, _ = data_manager.get_market_regime()

    dte = days_to_expiry(expiry_str)
    T_years = dte / 365.0

    short_leg_greeks = {"delta": 0, "gamma": 0, "theta": 0}
    long_leg_greeks = {"delta": 0, "gamma": 0, "theta": 0}
    
    stop_loss_price = None

    if current_price and dte > 0:
        if short_iv:
            short_leg_greeks = calculate_greeks("put", current_price, short_strike, T_years, 0.05, short_iv)
        if long_iv:
            long_leg_greeks = calculate_greeks("put", current_price, long_strike, T_years, 0.05, long_iv)
            
        # --- STOP LOSS CALCULATION (PATCHED) ---
        if short_iv and credit_received > 0:
            target_stop_loss_value = credit_received * 4.0
            
            # FIX: Use short_iv for BOTH legs to normalize skew.
            # This prevents bad OTM data from distorting the price target.
            calc_sigma = short_iv 

            stop_loss_price = find_stop_loss_price(
                target_value=target_stop_loss_value,
                option_type="put",
                short_K=short_strike,
                long_K=long_strike,
                T=T_years,
                r=0.05,
                short_sigma=calc_sigma, # Clean data
                long_sigma=calc_sigma   # Assumed flat skew
            )

    net_delta = (-1 * short_leg_greeks["delta"]) + (1 * long_leg_greeks["delta"])
    net_gamma = (-1 * short_leg_greeks["gamma"]) + (1 * long_leg_greeks["gamma"])
    net_theta = (-1 * short_leg_greeks["theta"]) + (1 * long_leg_greeks["theta"])

    profit_pct = None
    spread_val_pct = None
    if short_price is not None and long_price is not None and credit_received > 0:
        cost = short_price - long_price
        profit_pct = ((credit_received - cost) / credit_received) * 100
        spread_val_pct = (cost / credit_received) * 100

    rule_violations = { "other_rules": False, "iv_rule": False }
    notif_msg = ""
    notif_color = 15158332
    short_abs_delta = abs(short_leg_greeks["delta"]) 

    if not is_market_safe:
        rule_violations["other_rules"] = True
        notif_msg = "🚨 **CRASH ALERT**: SPY < 200 SMA."
    elif profit_pct is not None and profit_pct >= 60:
        notif_msg = f"✅ **Target Reached**: {profit_pct:.1f}% Profit"
        notif_color = 3066993 
    elif spread_val_pct is not None and spread_val_pct >= 400:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Stop Loss Hit**: Spread value > 400%."
        notif_color = 15105570
    elif dte <= 21:
        rule_violations["other_rules"] = True
        notif_msg = f"⚠️ **Exit Zone**: {dte} days left."
        notif_color = 15105570

    if notif_msg:
        last_sent_str = trade.get("last_alert_sent")
        should_send = True
        if last_sent_str:
            last_sent = datetime.fromisoformat(last_sent_str).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_sent < timedelta(hours=20): should_send = False
        if should_send:
            send_discord_alert(f"Position Alert: {ticker}", notif_msg, notif_color)
            trade["last_alert_sent"] = datetime.now(timezone.utc).isoformat()

    trade["contracts"] = contracts 
    
    trade["cached"] = {
        "current_price": current_price,
        "day_change_percent": day_change_pct,
        "price_history": price_history,
        "stop_loss_price": stop_loss_price,
        "short_option_price": short_price,
        "long_option_price": long_price,
        "abs_delta": short_abs_delta, 
        "net_delta": net_delta,
        "net_gamma": net_gamma,
        "net_theta": net_theta,
        "current_profit_percent": profit_pct,
        "spread_value_percent": spread_val_pct,
        "rule_violations": rule_violations,
        "spy_below_ma": not is_market_safe,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    today_str = date.today().isoformat()
    if "pnl_history" not in trade: trade["pnl_history"] = []
    existing = next((i for i in trade["pnl_history"] if i["date"] == today_str), None)
    if profit_pct is not None:
        if existing:
            existing["profit"] = profit_pct
            existing["dte"] = dte
            existing["net_theta"] = net_theta
        else:
            trade["pnl_history"].append({"date": today_str, "dte": dte, "profit": profit_pct, "net_theta": net_theta})

    return trade

def main():
    try:
        service = get_drive_service()
        lock = None
        if DriveLockManager:
            lock = DriveLockManager(service, FILE_ID)
            lock.acquire()
        try:
            trades = download_json(service)
            if not trades: return
            market_data = MarketDataManager()
            updated_trades = [update_trade(t, market_data) for t in trades]
            handle_heartbeat(updated_trades)
            upload_json(service, updated_trades)
            logger.info("Update complete.")
        finally:
            if lock: lock.release()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        raise

if __name__ == "__main__":
    main()
