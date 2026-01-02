import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests
from datetime import datetime, timedelta

# Import the Cloud Persistence
from persistence import (
    build_drive_service_from_session, 
    download_large_file_from_drive, 
    upload_large_file_to_drive,
    ensure_logged_in
)

st.set_page_config(layout="wide", page_title="Options Lab")

# --- AUTH CHECK (MANDATORY FOR CLOUD) ---
ensure_logged_in()

# --- CONFIG ---
SUCCESS_COLOR = "#00C853"
WARNING_COLOR = "#d32f2f"
BG_COLOR = '#0E1117'
TEXT_COLOR = '#FAFAFA'
GRID_COLOR = '#444444'

# --- CACHE (Ephemeral Cloud Storage) ---
CACHE_DIR = "options_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- MATPLOTLIB STYLE ---
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.3
})

# --- DATA PIPELINE ---
def download_from_web(url, local_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            prog_bar = st.progress(0)
            status = st.empty()
            
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        prog_bar.progress(min(downloaded / total_size, 1.0))
                        status.caption(f"Fetching: {int(downloaded/1024/1024)}MB")
            
            prog_bar.empty()
            status.empty()
            return True
    except Exception as e:
        st.error(f"Web source failed: {e}")
        return False

@st.cache_data(ttl=3600*24, show_spinner=False)
def load_data_pipeline(ticker, _drive_service):
    """
    1. Check Ephemeral Cache
    2. Check Google Drive
    3. Download from Web -> Upload to Drive
    """
    filename = f"{ticker}_options.parquet"
    local_path = os.path.join(CACHE_DIR, filename)
    web_url = f"https://static.philippdubach.com/data/options/{ticker}/options.parquet"
    
    source = "Local Cache"

    if not os.path.exists(local_path):
        st.info(f"Searching Google Drive for {ticker}...")
        drive_bar = st.progress(0)
        
        # Try downloading from Drive
        if download_large_file_from_drive(_drive_service, filename, local_path, drive_bar):
            source = "Google Drive"
            st.toast(f"Restored {ticker} from Drive", icon="☁️")
        else:
            # Fallback to Web
            st.warning(f"Not in Drive. Downloading {ticker} from source... (This happens once)")
            if download_from_web(web_url, local_path):
                source = "Web Source"
                st.info("Backing up to Drive for next time...")
                up_bar = st.progress(0)
                upload_large_file_to_drive(_drive_service, local_path, filename, up_bar)
                up_bar.empty()
                st.toast("Backup Created!", icon="💾")
            else:
                return None, None
        drive_bar.empty()

    # Load into Pandas (Pruned Columns)
    try:
        cols = ['quote_date', 'expire_date', 'strike', 'option_type', 'delta', 'bid', 'ask']
        df = pd.read_parquet(local_path, columns=cols)
        df = df[df['option_type'] == 'P'] # Puts only
        
        # Optimizations
        df['quote_date'] = pd.to_datetime(df['quote_date'])
        df['expire_date'] = pd.to_datetime(df['expire_date'])
        df['dte'] = (df['expire_date'] - df['quote_date']).dt.days
        df['mid'] = (df['bid'] + df['ask']) / 2
        return df, source
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None, None

# --- UI ---
st.title("Options Lab (Cloud Edition)")

drive_service = build_drive_service_from_session()
if not drive_service:
    st.error("Auth Error: No Drive Service. Refresh page.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1: ticker = st.selectbox("Ticker", ["QQQ", "IWM", "SPY"])
with col2: e_delta = st.number_input("Exit Delta", 0.1, 1.0, 0.4)
with col3: e_stop = st.number_input("Stop %", 100, 500, 200)
with col4: e_dte = st.number_input("Exit DTE", 0, 30, 14)

if st.button("Run Simulation", type="primary"):
    df, src = load_data_pipeline(ticker, drive_service)
    
    if df is not None:
        st.success(f"Loaded {len(df):,} rows from {src}")
        
        # --- SIMPLE BACKTEST LOOP (Optimized for Streamlit) ---
        trades = []
        dates = sorted(df['quote_date'].unique())
        # Mondays only
        mondays = [d for d in dates if d.dayofweek == 0]
        
        bar = st.progress(0)
        
        for i, date in enumerate(mondays):
            if i % 10 == 0: bar.progress(i / len(mondays))
            
            # 1. Filter: 45 DTE
            day = df[(df['quote_date'] == date) & (df['dte'].between(35, 50))]
            if day.empty: continue
            
            # 2. Entry: 20 Delta Put
            day = day.copy()
            day['d_diff'] = abs(abs(day['delta']) - 0.20)
            short = day.loc[day['d_diff'].idxmin()]
            
            # 3. Long: $5 wide
            longs = day[abs(day['strike'] - (short['strike'] - 5)) < 0.5]
            if longs.empty: continue
            long = longs.iloc[0]
            
            credit = short['mid'] - long['mid']
            if credit < 0.50: continue
            
            # 4. Outcome
            res = {'date': date, 'pnl': credit, 'reason': 'Held'}
            
            # Look forward
            future = df[(df['expire_date'] == short['expire_date']) & (df['quote_date'] > date)]
            
            for f_date, grp in future.groupby('quote_date'):
                s_cur = grp[grp['strike'] == short['strike']]
                l_cur = grp[grp['strike'] == long['strike']]
                
                if s_cur.empty: continue
                val = s_cur.iloc[0]['mid'] - (l_cur.iloc[0]['mid'] if not l_cur.empty else 0)
                
                # Exit logic
                if (val/credit) * 100 > e_stop:
                    res.update({'pnl': credit-val, 'reason': 'Stop'}); break
                if (short['expire_date'] - f_date).days < e_dte:
                    res.update({'pnl': credit-val, 'reason': 'DTE'}); break
                if abs(s_cur.iloc[0]['delta']) > e_delta:
                    res.update({'pnl': credit-val, 'reason': 'Delta'}); break
            
            trades.append(res)
            
        bar.empty()
        
        if trades:
            res_df = pd.DataFrame(trades)
            res_df['cum'] = res_df['pnl'].cumsum() * 100
            st.metric("Total P&L", f"${res_df['cum'].iloc[-1]:,.2f}")
            st.line_chart(res_df.set_index('date')['cum'])
            st.dataframe(res_df)
        else:
            st.warning("No trades found.")
