import streamlit as st
import json
import io
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

# --- GOOGLE AUTH IMPORTS ---
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
import google.auth.exceptions # IMPORT ADDED FOR ERROR HANDLING
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

LOCAL_TOKEN_FILE = "google_oauth_token.json"

# ----------------------------- AUTHENTICATION (ST.SECRETS) -----------------------------
def _get_oauth_config():
    if "google_oauth" not in st.secrets:
        st.error("Missing 'google_oauth' in Streamlit Secrets.")
        st.stop()
    return st.secrets["google_oauth"]

def _get_web_client_config(cfg):
    return {
        "web": {
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uris": [cfg["redirect_uri"]],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def _credentials_to_dict(creds):
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

def _save_local_token(creds_dict):
    try:
        with open(LOCAL_TOKEN_FILE, "w") as f:
            json.dump(creds_dict, f)
    except: pass

def _load_local_token():
    try:
        if os.path.exists(LOCAL_TOKEN_FILE):
            with open(LOCAL_TOKEN_FILE, "r") as f:
                return json.load(f)
    except: return None

def get_creds_from_session():
    """
    Gets valid credentials. Handles REFRESH errors by forcing logout.
    """
    # 1. Try Session
    creds_dict = st.session_state.get("credentials")
    
    # 2. Try Local Cache (if session empty)
    if not creds_dict:
        creds_dict = _load_local_token()
        
    if not creds_dict:
        return None
    
    try:
        creds = OAuthCredentials(**creds_dict)
        
        # 3. Check Expiry & Refresh
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save new token
                new_dict = _credentials_to_dict(creds)
                st.session_state["credentials"] = new_dict
                _save_local_token(new_dict)
            except google.auth.exceptions.RefreshError:
                # CRITICAL FIX: If refresh fails (401), kill the session immediately
                st.warning("Session expired. Please sign in again.")
                st.session_state.pop("credentials", None)
                if os.path.exists(LOCAL_TOKEN_FILE):
                    os.remove(LOCAL_TOKEN_FILE)
                return None
                
        return creds
    except Exception as e:
        # If anything else breaks in auth, return None to force re-login
        print(f"Auth Error: {e}")
        return None

def ensure_logged_in():
    """
    Blocking call. If not logged in, shows Sign-In button and stops script.
    """
    # 1. Check for valid session creds
    if get_creds_from_session():
        return

    # 2. Check for OAuth Callback (Code in URL)
    if "code" in st.query_params:
        try:
            cfg = _get_oauth_config()
            client_config = _get_web_client_config(cfg)
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                redirect_uri=cfg["redirect_uri"]
            )
            flow.fetch_token(code=st.query_params["code"])
            creds_dict = _credentials_to_dict(flow.credentials)
            
            # Store
            st.session_state["credentials"] = creds_dict
            _save_local_token(creds_dict)
            
            # Clear URL and Rerun
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.stop()

    # 3. Show Login Button
    cfg = _get_oauth_config()
    client_config = _get_web_client_config(cfg)
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=cfg["redirect_uri"]
    )
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    
    st.markdown("### 🔒 Security Check")
    st.warning("Please sign in to access your Cloud Dashboard.")
    st.markdown(f"[**👉 Sign in with Google**]({auth_url})", unsafe_allow_html=True)
    st.stop()

def logout():
    st.session_state.clear()
    if os.path.exists(LOCAL_TOKEN_FILE):
        os.remove(LOCAL_TOKEN_FILE)
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
    try:
        q = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
        files = results.get('files', [])
        return files[0]['id'] if files else None
    except: return None

def _get_or_create_folder(service, folder_name):
    try:
        q = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        files = service.files().list(q=q, spaces='drive', fields='files(id)').execute().get('files', [])
        if files: return files[0]['id']
        
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        return service.files().create(body=metadata, fields='id').execute().get('id')
    except: return None

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
    
    clean_trades = []
    for t in trades:
        ct = t.copy()
        for k, v in ct.items():
            if isinstance(v, (date, datetime)): ct[k] = v.isoformat()
        clean_trades.append(ct)
        
    content = json.dumps(clean_trades, indent=2)
    media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='application/json')
    
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    try:
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            service.files().create(body={'name': DRIVE_FILE_NAME}, media_body=media).execute()
        return True
    except: return False

# ----------------------------- LOGGING (SHEETS) -----------------------------
def log_completed_trade(service, trade_data):
    if not service: return False
    try:
        sheets_service = build_sheets_service_from_session()
        q = f"name = '{SPREADSHEET_NAME}' and mimeType = 'application/vnd.google-apps.spreadsheet'"
        files = service.files().list(q=q).execute().get('files', [])
        if files:
            spreadsheet_id = files[0]['id']
        else:
            ss = sheets_service.spreadsheets().create(body={'properties': {'title': SPREADSHEET_NAME}}).execute()
            spreadsheet_id = ss['spreadsheetId']

        ss_meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        if not any(s['properties']['title'] == "TradeLog" for s in ss_meta['sheets']):
            sheets_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={
                "requests": [{"addSheet": {"properties": {"title": "TradeLog"}}}]
            }).execute()
            headers = [["Ticker", "Entry", "Exit", "Strike_Short", "Strike_Long", "Credit", "Debit", "Contracts", "PnL", "Notes"]]
            sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id, range="TradeLog!A1",
                valueInputOption="RAW", body={"values": headers}
            ).execute()

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
