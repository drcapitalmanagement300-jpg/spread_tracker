import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si
import google.generativeai as genai
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Spread Sniper")

# 1. Initialize Gemini (Optional)
try:
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        pass
except Exception:
    pass

# 2. Universe (Liquid S&P 500 Proxies)
LIQUID_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "AMZN", "MSFT", 
    "GOOGL", "META", "NFLX", "BAC", "JPM", "XOM", "CVX", "DIS", "BA", "PLTR", "COIN"
]

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Metric Typography */
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
    .metric-value { font-size: 16px; font-weight: 700; color: #FFF; }
    
    /* Pills & Badges */
    .price-pill-red { background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .price-pill-green { background-color: rgba(0, 200, 100, 0.15); color: #00c864; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .strategy-badge { border: 1px solid #d4ac0d; color: #d4ac0d; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase; }
    
    /* Footer Box */
    .roc-box { background-color: rgba(0, 255, 127, 0.05); border: 1px solid rgba(0, 255, 127, 0.2); border-radius: 6px; padding: 8px; text-align: center; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)


# --- MATH & CALCULATORS ---

def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)
    return sigma - fx / vega

def get_implied_volatility(price, strike, time_to_exp, market_price, risk_free_rate=0.045):
    sigma = 0.5 
    try:
        for i in range(50):
            diff = newton_vol_put(price, strike, time_to_exp, market_price, risk_free_rate, sigma)
            sigma = diff
            if abs(diff - sigma) < 1e-5: break
        return abs(sigma)
    except: return 0.0

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # --- PROXY: REALIZED VOLATILITY RANK ---
        # 1. Calculate 30-Day Rolling Volatility (Annualized)
        hist['Returns'] = hist['Close'].pct_change()
        hist['HV'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252) * 100
        
        # 2. Rank current HV against the last 52 weeks of HV
        rank = 50
        if not hist['HV'].empty:
            mn, mx = hist['HV'].min(), hist['HV'].max()
            if mx != mn: 
                rank = ((hist['HV'].iloc[-1] - mn) / (mx - mn)) * 100

        return {
            "price": current_price,
            "change_pct": change_pct,
            "rank": rank,
            "hist": hist
        }
    except: return None

def find_credit_spread(stock_obj, current_price, target_dte=30, width=5.0):
    try:
        exps = stock_obj.options
        if not exps: return None
        target_date = datetime.now() + timedelta(days=target_dte)
        best_exp = min(exps, key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - target_date).days))
        dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
        if dte < 7: return None 

        chain = stock_obj.option_chain(best_exp)
        puts = chain.puts
        
        # Target 30 Delta (Approx ~4% OTM for average vol)
        target_strike = current_price * 0.96
        short_leg = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]
        if short_leg.empty: return None
        
        short_strike = short_leg.iloc[0]['strike']
        short_mid = (short_leg.iloc[0]['bid'] + short_leg.iloc[0]['ask']) / 2
        
        long_leg = puts.iloc[(puts['strike'] - (short_strike - width)).abs().argsort()[:1]]
        if long_leg.empty: return None
        long_strike = long_leg.iloc[0]['strike']
        
        # Strict Width Check (Allowing small variance for weird strikes)
        if abs((short_strike - long_strike) - width) > 1.0: return None

        credit = short_leg.iloc[0]['bid'] - long_leg.iloc[0]['ask']
        max_loss = (short_strike - long_strike) - credit
        
        # Calculate Spread IV (Live)
        iv = get_implied_volatility(current_price, short_strike, dte/365, short_mid) * 100

        return {
            "expiration": best_exp, "dte": dte, "short": short_strike, "long": long_strike,
            "credit": credit, "max_loss": max_loss, "iv": iv,
            "roi": (credit / max_loss) *
