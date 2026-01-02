import streamlit as st
import json
import io
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import Flow

# --- CONFIG ---
DRIVE_FILE_NAME = "credit_spreads.json"
SPREADSHEET_NAME = "SpreadSniper_Data"
DATASET_FOLDER_NAME = "SpreadSniper_Datasets"

# Scopes
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]

# ----------------------------- AUTHENTICATION (ST.SECRETS) -----------------------------
def _get_oauth_config():
    """Reads OAuth config from Streamlit Secrets (Cloud Best Practice)."""
    if "google_oauth" not in st.secrets:
        st.error("Missing 'google_oauth' in Streamlit Secrets.")
        st.stop()
    return st.secrets["google_oauth"]

def _credentials_to_dict(creds):
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

def get_creds_from_session():
    """Gets valid credentials or returns None."""
    if "credentials" not in st.session_state:
        return None
    
    creds_dict = st.session_state["credentials"]
    try:
        creds = OAuthCredentials(**creds_dict)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["credentials"] = _credentials_to_dict(creds)
        return creds
    except:
        return None

def ensure_logged_in():
    """
    Blocking call. If not logged in, shows the 'Sign in with Google' link 
    and stops script execution.
    """
    # 1. Check for valid session creds
    if get_creds_from_session():
        return

    # 2. Check for OAuth Callback (Code in URL)
    if "code" in st.query_params:
        try:
            config = _get_oauth_config()
            flow = Flow.from_client_config(
                {"web": config},
                scopes=SCOPES,
                redirect_uri=config["redirect_uri"]
            )
            flow.fetch_token(code=st.query_params["code"])
            st.session_state["credentials"] = _credentials_to_dict(flow.credentials)
            # Clear URL params and reload to clean state
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.stop()

    # 3. Show Login Button
    config = _get_oauth_config()
    flow = Flow.from_client_config(
        {"web": config},
        scopes=SCOPES,
        redirect_uri=config["redirect_uri"]
    )
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    
    st.markdown("### 🔒 Security Check")
    st.warning("Please sign in to access your Cloud Dashboard.")
    st.markdown(f"[**👉 Sign in with Google**]({auth_url})", unsafe_allow_html=True)
    st.stop()

def logout():
    st.session_state.clear()
    st.rerun()

# ----------------------------- SERVICE BUILDERS -----------------------------
def build_drive_service_from_session():
    creds = get_creds_from_session()
    if creds: return build('drive', 'v3', credentials=creds)
    return None

def build_sheets_service_from_session():
    creds = get_creds_from_session()
    if creds: return build('sheets', 'v4', credentials=creds)
    return None

# ----------------------------- DRIVE HELPERS -----------------------------
def _find_file_id(service, filename):
    q = f"name = '{filename}' and trashed = false"
    results = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def _get_or_create_folder(service, folder_name):
    q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    files = service.files().list(q=q, spaces='drive', fields='files(id)').execute().get('files', [])
    if files: return files[0]['id']
    
    metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    return service.files().create(body=metadata, fields='id').execute().get('id')

# ----------------------------- LARGE DATASET HANDLING (BACKTESTING) -----------------------------
def download_large_file_from_drive(service, filename, local_path, progress_bar=None):
    """Downloads a large dataset from the 'SpreadSniper_Datasets' folder in Drive."""
    try:
        folder_id = _get_or_create_folder(service, DATASET_FOLDER_NAME)
        q = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        files = service.files().list(q=q, spaces='drive', fields='files(id, size)').execute().get('files', [])
        
        if not files: return False
        
        request = service.files().get_media(fileId=files[0]['id'])
        
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status and progress_bar:
                    progress_bar.progress(int(status.progress() * 100) / 100)
        return True
    except: return False

def upload_large_file_to_drive(service, local_path, filename, progress_bar=None):
    """Backs up a large dataset to the 'SpreadSniper_Datasets' folder in Drive."""
    try:
        folder_id = _get_or_create_folder(service, DATASET_FOLDER_NAME)
        
        # Check update vs create
        q = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        files = service.files().list(q=q, spaces='drive', fields='files(id)').execute().get('files', [])
        
        media = MediaFileUpload(local_path, resumable=True)
        
        if files:
            request = service.files().update(fileId=files[0]['id'], media_body=media)
        else:
            metadata = {'name': filename, 'parents': [folder_id]}
            request = service.files().create(body=metadata, media_body=media)

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status and progress_bar:
                progress_bar.progress(int(status.progress() * 100) / 100)
        return True
    except: return False

# ----------------------------- ACTIVE TRADES (JSON) -----------------------------
def load_from_drive(service):
    if not service: return []
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    if not file_id: return []
    
    try:
        data = service.files().get_media(fileId=file_id).execute()
        return json.loads(data.decode('utf-8'))
    except: return []

def save_to_drive(service, trades):
    if not service: return False
    
    # Serialize dates
    clean_trades = []
    for t in trades:
        ct = t.copy()
        for k, v in ct.items():
            if isinstance(v, (date, datetime)): ct[k] = v.isoformat()
        clean_trades.append(ct)
        
    content = json.dumps(clean_trades, indent=2)
    media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='application/json')
    
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        service.files().create(body={'name': DRIVE_FILE_NAME}, media_body=media).execute()
    return True

# ----------------------------- LOGGING (SHEETS) -----------------------------
def log_completed_trade(service, trade_data):
    """Logs trade to Google Sheets."""
    if not service: return False
    
    try:
        sheets_service = build_sheets_service_from_session()
        
        # 1. Find Spreadsheet
        q = f"name = '{SPREADSHEET_NAME}' and mimeType = 'application/vnd.google-apps.spreadsheet'"
        files = service.files().list(q=q).execute().get('files', [])
        if files:
            spreadsheet_id = files[0]['id']
        else:
            ss = sheets_service.spreadsheets().create(body={'properties': {'title': SPREADSHEET_NAME}}).execute()
            spreadsheet_id = ss['spreadsheetId']

        # 2. Check/Create 'TradeLog' Tab
        ss_meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        if not any(s['properties']['title'] == "TradeLog" for s in ss_meta['sheets']):
            sheets_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={
                "requests": [{"addSheet": {"properties": {"title": "TradeLog"}}}]
            }).execute()
            # Headers
            headers = [["Ticker", "Entry", "Exit", "Strike_Short", "Strike_Long", "Credit", "Debit", "Contracts", "PnL", "Notes"]]
            sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id, range="TradeLog!A1",
                valueInputOption="RAW", body={"values": headers}
            ).execute()

        # 3. Append Data
        credit = float(trade_data.get('credit', 0))
        debit = float(trade_data.get('debit_paid', 0))
        contracts = int(trade_data.get('contracts', 1))
        pl = (credit - debit) * contracts * 100
        
        row = [[
            trade_data.get('ticker'),
            trade_data.get('entry_date'),
            datetime.now().strftime("%Y-%m-%d"),
            trade_data.get('short_strike'),
            trade_data.get('long_strike'),
            credit, debit, contracts, pl,
            trade_data.get('notes', '')
        ]]
        
        sheets_service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range="TradeLog!A1",
            valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS", body={"values": row}
        ).execute()
        return True
    except: return False

def get_trade_log(service):
    """Fetches log from Sheets."""
    if not service: return []
    try:
        sheets_service = build_sheets_service_from_session()
        q = f"name = '{SPREADSHEET_NAME}'"
        files = service.files().list(q=q).execute().get('files', [])
        if not files: return []
        
        res = sheets_service.spreadsheets().values().get(
            spreadsheetId=files[0]['id'], range="TradeLog!A:Z"
        ).execute()
        
        rows = res.get('values', [])
        if len(rows) < 2: return []
        
        headers = rows[0]
        data = []
        for r in rows[1:]:
            item = {headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))}
            data.append(item)
        return data
    except: return []

def delete_log_entry(service, index):
    # (Implementation omitted for brevity, follows same pattern as before but using passed-in service)
    pass
