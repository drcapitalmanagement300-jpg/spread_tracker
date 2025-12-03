# Full integrated Streamlit app with Google OAuth + Google Drive JSON persistence

import streamlit as st
from datetime import date, datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
import json
import io
from typing import List, Dict, Any, Optional

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("Options Spread Monitor")

# ------------------- STATE INIT ---------------------
def init_state():
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "google_creds" not in st.session_state:
        st.session_state.google_creds = None

init_state()

# ---------------- GOOGLE DRIVE HELPERS -----------------

def get_drive_client():
    creds = st.session_state.get("google_creds", None)
    if not creds:
        return None
    return build("drive", "v3", credentials=creds)


def load_trades_from_drive():
    drive = get_drive_client()
    if drive is None:
        return None

    try:
        response = drive.files().list(
            q="name='trades.json' and mimeType='application/json'",
            spaces='drive'
        ).execute()
        files = response.get("files", [])

        if not files:
            return None

        file_id = files[0]["id"]

        request = drive.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        fh.seek(0)
        data = json.loads(fh.read().decode("utf-8"))
        return data

    except Exception as e:
        st.error(f"Failed to load trades: {e}")
        return None


def save_trades_to_drive(trades):
    drive = get_drive_client()
    if drive is None:
        return

    json_bytes = json.dumps(trades, indent=2).encode("utf-8")
    fh = io.BytesIO(json_bytes)

    response = drive.files().list(
        q="name='trades.json' and mimeType='application/json'",
        spaces='drive'
    ).execute()
    files = response.get("files", [])

    media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=True)

    try:
        if files:
            file_id = files[0]["id"]
            drive.files().update(fileId=file_id, media_body=media).execute()
        else:
            drive.files().create(
                body={"name": "trades.json", "mimeType": "application/json"},
                media_body=media
            ).execute()
    except Exception as e:
        st.error(f"Failed to save trades: {e}")


# ------------------- GOOGLE OAUTH ----------------------

# This is where Streamlit stores tokens automatically
from streamlit_oauth import OAuth2Component

client_id = st.secrets["gcp_oauth"]["client_id"]
client_secret = st.secrets["gcp_oauth"]["client_secret"]
redirect_uri = st.secrets["gcp_oauth"]["redirect_uri"]

oauth = OAuth2Component(
    client_id,
    client_secret,
    redirect_uri,
    authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
)

# Request Drive scope (read/write)
scope = "https://www.googleapis.com/auth/drive.file"

token = oauth.authorize_button("Login with Google", scope=scope)

if token is not None and st.session_state.google_creds is None:
    st.session_state.google_creds = Credentials(
        token["access_token"],
        refresh_token=token.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=[scope]
    )
    st.rerun()

# If logged in, load trades
if st.session_state.google_creds:
    loaded = load_trades_from_drive()
    if loaded is not None:
        st.session_state.trades = loaded

# ------------------- TRADE INPUT ---------------------
st.subheader("Add New Trade")
ticker = st.text_input("Ticker")
entry_date = st.date_input("Entry Date", value=date.today())
credit = st.number_input("Credit Received", min_value=0.0, step=0.01)
short_strike = st.number_input("Short Strike", step=0.5)
long_strike = st.number_input("Long Strike", step=0.5)

if st.button("Add Trade"):
    trade = {
        "ticker": ticker.upper(),
        "entry_date": str(entry_date),
        "credit": credit,
        "short": short_strike,
        "long": long_strike,
    }
    st.session_state.trades.append(trade)
    save_trades_to_drive(st.session_state.trades)
    st.rerun()

# ------------------- TRADE TABLE ----------------------
st.subheader("Current Trades")

for i, t in enumerate(st.session_state.trades):
    st.write(t)
    if st.button(f"Delete {i}"):
        st.session_state.trades.pop(i)
        save_trades_to_drive(st.session_state.trades)
        st.rerun()
```python
import streamlit as st
from datetime import date, datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
import json
import io
from typing import List, Dict, Any, Optional

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")

# ---------------------- GOOGLE DRIVE AUTH ----------------------
# Loads credentials from Streamlit secrets and returns a Drive service
@st.cache_resource
def get_gdrive_service():
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/drive"])
    service = build("drive", "v3", credentials=creds)
    return service

# Upload JSON data to Drive
# If file exists, it is replaced (same fileId)
def upload_json_to_drive(service, file_id: str, data: dict):
    json_bytes = json.dumps(data).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(json_bytes), mimetype="application/json", resumable=False)
    service.files().update(fileId=file_id, media_body=media).execute()

# Download JSON from Drive
# Returns {} if file missing or corrupted

def download_json_from_drive(service, file_id: str) -> dict:
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        return json.loads(fh.read().decode("utf-8"))
    except Exception:
        return {}

# ---------------------- STATE MANAGEMENT ----------------------
FILE_ID = st.secrets["drive"]["json_file_id"]
service = get_gdrive_service()

# Load stored trades
stored_data = download_json_from_drive(service, FILE_ID)
if "trades" not in stored_data:
    stored_data["trades"] = []

# Put the storage into session_state
if "trades" not in st.session_state:
    st.session_state.trades = stored_data["trades"]

# ---------------------- UI ----------------------
st.title("Credit Spread Monitor — Google Drive Synced")

st.subheader("Add New Trade")
ticker = st.text_input("Ticker")
open_date = st.date_input("Open Date", value=date.today())
credit = st.number_input("Credit Received", min_value=0.0, format="%.2f")
short_strike = st.number_input("Short Strike", min_value=0.0)
long_strike = st.number_input("Long Strike", min_value=0.0)
exp_date = st.date_input("Expiration Date")

if st.button("Add Trade"):
    trade = {
        "ticker": ticker,
        "open_date": str(open_date),
        "credit": credit,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "exp": str(exp_date),
    }
    st.session_state.trades.append(trade)

    # Save back to Drive
    upload_json_to_drive(service, FILE_ID, {"trades": st.session_state.trades})
    st.success("Trade saved to Google Drive!")

st.subheader("Saved Trades")
st.dataframe(st.session_state.trades)
```
