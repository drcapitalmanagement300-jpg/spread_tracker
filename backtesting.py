import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
from datetime import datetime, timedelta

# Import the new Drive tools
from persistence import (
    build_drive_service_from_session, 
    download_large_file_from_drive, 
    upload_large_file_to_drive
)

st.set_page_config(layout="wide", page_title="Backtesting")

# --- CACHE SETUP ---
CACHE_DIR = "options_data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- HELPER: WEB DOWNLOAD ---
def download_from_web(url, local_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            progress_text = st.empty()
            progress_bar = st.progress(0)
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
                        progress_text.caption(f"Downloading from Web: {downloaded/1024/1024:.1f} MB")
            progress_bar.empty()
            progress_text.empty()
            return True
    except Exception as e:
        st.error(f"Web Download failed: {e}")
        return False

# --- DATA LOADER PIPELINE ---
@st.cache_data(ttl=3600*24, show_spinner=False)
def load_data_pipeline(ticker, _drive_service):
    filename = f"{ticker}_options.parquet"
    local_path = os.path.join(CACHE_DIR, filename)
    web_url = f"https://static.philippdubach.com/data/options/{ticker}/options.parquet"
    
    # 1. Check Local Cache (Fastest)
    if os.path.exists(local_path):
        # Proceed to load
        pass
    
    # 2. Check Google Drive (Backup)
    elif _drive_service and download_large_file_from_drive(_drive_service, filename, local_path):
        st.toast(f"Restored {ticker} data from Google Drive!", icon="☁️")
        
    # 3. Download from Web (Last Resort)
    else:
        st.info(f"Data not found locally or on Drive. Downloading {ticker} from source...")
        if download_from_web(web_url, local_path):
            st.success("Download complete.")
            
            # 4. Upload to Drive for next time
            if _drive_service:
                st.info("Backing up to Google Drive...")
                upload_large_file_to_drive(_drive_service, local_path, filename)
                st.toast("Backed up to Drive!", icon="💾")
        else:
            return None

    # 5. Load into DataFrame (Pruned)
    try:
        columns = ['quote_date', 'expire_date', 'strike', 'option_type', 'delta', 'bid', 'ask']
        df = pd.read_parquet(local_path, columns=columns)
        df = df[df['option_type'] == 'P']
        df['quote_date'] = pd.to_datetime(df['quote_date'])
        df['expire_date'] = pd.to_datetime(df['expire_date'])
        df['dte'] = (df['expire_date'] - df['quote_date']).dt.days
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        return df
    except Exception as e:
        st.error(f"File corrupt: {e}")
        if os.path.exists(local_path): os.remove(local_path)
        return None

# --- BACKTEST LOGIC (Same as before) ---
def run_backtest(df, entry_delta, exit_delta, exit_stop, exit_dte):
    trades = []
    dates = sorted(df['quote_date'].unique())
    entry_dates = [d for d in dates if pd.Timestamp(d).dayofweek == 0] # Mondays
    
    bar = st.progress(0)
    for i, date in enumerate(entry_dates):
        if i % 5 == 0: bar.progress((i+1)/len(entry_dates))
        
        # Entry
        daily = df[(df['quote_date'] == date) & (df['dte'] >= 30) & (df['dte'] <= 50)]
        if daily.empty: continue
        
        # Find Short
        daily = daily.copy()
        daily['d_diff'] = abs(abs(daily['delta']) - abs(entry_delta))
        short = daily.loc[daily['d_diff'].idxmin()]
        
        # Find Long ($5 wide)
        longs = daily[abs(daily['strike'] - (short['strike'] - 5)) < 0.5]
        if longs.empty: continue
        long = longs.iloc[0]
        
        credit = short['mid_price'] - long['mid_price']
        if credit < 0.50: continue
        
        outcome = {'date': date, 'pnl': credit, 'reason': 'Held'}
        
        # Walk forward
        future = df[(df['expire_date'] == short['expire_date']) & (df['quote_date'] > date)]
        for fd, grp in future.groupby('quote_date'):
            s_curr = grp[grp['strike'] == short['strike']]
            l_curr = grp[grp['strike'] == long['strike']]
            if s_curr.empty: continue
            
            val = s_curr.iloc[0]['mid_price'] - (l_curr.iloc[0]['mid_price'] if not l_curr.empty else 0)
            pct = (val / credit) * 100
            dte_now = (short['expire_date'] - fd).days
            delta_now = abs(s_curr.iloc[0]['delta'])
            
            if dte_now < exit_dte or delta_now > exit_delta or pct > exit_stop or pct < 50:
                outcome['pnl'] = credit - val
                outcome['reason'] = "Exit Triggered"
                break
        trades.append(outcome)
        
    bar.empty()
    return pd.DataFrame(trades)

# --- UI ---
st.title("Options Lab (Drive-Enabled)")
drive_service = build_drive_service_from_session()

if not drive_service:
    st.warning("⚠️ Google Drive not connected. Data will only be saved locally (Temporary).")

c1, c2, c3, c4 = st.columns(4)
with c1: ticker = st.selectbox("Ticker", ["QQQ", "SPY", "IWM"])
with c2: e_delta = st.number_input("Exit Delta", 0.1, 1.0, 0.40)
with c3: e_stop = st.number_input("Stop %", 100, 500, 200)
with c4: e_dte = st.number_input("Exit DTE", 0, 30, 14)

if st.button("Run Simulation"):
    df = load_data_pipeline(ticker, drive_service)
    if df is not None:
        res = run_backtest(df, 0.20, e_delta, e_stop, e_dte)
        if not res.empty:
            st.metric("Total P&L", f"${res['pnl'].sum()*100:,.2f}")
            st.line_chart(res['pnl'].cumsum())
        else:
            st.info("No trades found.")
