import streamlit as st
import json
import io
import os
import logging
from datetime import datetime, date

# --- GOOGLE AUTH IMPORTS ---
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
import google.auth.exceptions
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import Flow

# Import Lock Manager
try:
    from google_drive import DriveLockManager
except ImportError:
    DriveLockManager = None

# --- CONFIG ---
DRIVE_FILE_NAME = "credit_spreads.json"
SPREADSHEET_NAME = "SpreadSniper_Data"
SHEET_TITLE = "TradeLog"
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
LOCAL_TOKEN_FILE = "google_oauth_token.json"

# ----------------------------- AUTHENTICATION -----------------------------
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
    # 1. Try Session
    creds_dict = st.session_state.get("credentials")
    
    # 2. Try Local Cache
    if not creds_dict:
        creds_dict = _load_local_token()
        
    if not creds_dict: return None
    
    try:
        creds = OAuthCredentials(**creds_dict)
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                new_dict = _credentials_to_dict(creds)
                st.session_state["credentials"] = new_dict
                _save_local_token(new_dict)
            except google.auth.exceptions.RefreshError:
                st.warning("Session expired. Please sign in again.")
                st.session_state.pop("credentials", None)
                if os.path.exists(LOCAL_TOKEN_FILE):
                    os.remove(LOCAL_TOKEN_FILE)
                return None
        return creds
    except: return None

def ensure_logged_in():
    if get_creds_from_session(): return

    if "code" in st.query_params:
        try:
            cfg = _get_oauth_config()
            flow = Flow.from_client_config(
                _get_web_client_config(cfg),
                scopes=SCOPES,
                redirect_uri=cfg["redirect_uri"]
            )
            flow.fetch_token(code=st.query_params["code"])
            creds_dict = _credentials_to_dict(flow.credentials)
            st.session_state["credentials"] = creds_dict
            _save_local_token(creds_dict)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.stop()

    cfg = _get_oauth_config()
    flow = Flow.from_client_config(
        _get_web_client_config(cfg),
        scopes=SCOPES,
        redirect_uri=cfg["redirect_uri"]
    )
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    
    st.markdown("### 🔒 Security Check")
    st.warning("Please sign in to access your Cloud Dashboard.")
    st.markdown(f"[**Sign in with Google**]({auth_url})", unsafe_allow_html=True)
    st.stop()

def logout():
    st.session_state.clear()
    if os.path.exists(LOCAL_TOKEN_FILE):
        os.remove(LOCAL_TOKEN_FILE)
    st.rerun()

# ----------------------------- SERVICES -----------------------------
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

# ----------------------------- JSON STORAGE -----------------------------
def load_from_drive(service):
    if not service: return []
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    if not file_id: return []
    try:
        data = service.files().get_media(fileId=file_id).execute()
        return json.loads(data.decode('utf-8'))
    except: return []

def save_to_drive(service, trades):
    """
    Saves trades to Drive using DriveLockManager to prevent race conditions.
    """
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
    
    lock = None
    if DriveLockManager and file_id:
        lock = DriveLockManager(service, file_id)
        try:
            lock.acquire()
        except Exception as e:
            print(f"Warning: Could not acquire lock, proceeding: {e}")
            lock = None

    try:
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            service.files().create(body={'name': DRIVE_FILE_NAME}, media_body=media).execute()
        return True
    except Exception as e:
        print(f"Save Error: {e}")
        return False
    finally:
        if lock: lock.release()

# ----------------------------- SHEETS LOGGING (THE WRITER) -----------------------------
def _get_spreadsheet_id(service, sheets_service):
    # 1. Find existing
    q = f"name = '{SPREADSHEET_NAME}' and mimeType = 'application/vnd.google-apps.spreadsheet'"
    files = service.files().list(q=q).execute().get('files', [])
    if files: return files[0]['id']
    
    # 2. Create new
    ss = sheets_service.spreadsheets().create(body={'properties': {'title': SPREADSHEET_NAME}}).execute()
    return ss['spreadsheetId']

def _ensure_sheet_exists(sheets_service, spreadsheet_id):
    meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_id = None
    for s in meta['sheets']:
        if s['properties']['title'] == SHEET_TITLE:
            sheet_id = s['properties']['sheetId']
            break
            
    # FIXED: Columns match Log Review expectations exactly
    header_row = ["Ticker", "Entry Date", "Exit Date", "Short Strike", "Long Strike", "Contracts", "Credit", "Debit", "PnL", "Earnings Date", "Notes"]
            
    if sheet_id is None:
        # Create sheet
        req = {"addSheet": {"properties": {"title": SHEET_TITLE}}}
        res = sheets_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": [req]}).execute()
        sheet_id = res['replies'][0]['addSheet']['properties']['sheetId']
        
        # Add Headers
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id, range=f"{SHEET_TITLE}!A1",
            valueInputOption="RAW", body={"values": [header_row]}
        ).execute()
        
    return sheet_id

def log_completed_trade(service, trade_data):
    if not service: return False
    try:
        sheets_service = build_sheets_service_from_session()
        spreadsheet_id = _get_spreadsheet_id(service, sheets_service)
        _ensure_sheet_exists(sheets_service, spreadsheet_id)

        # Data extraction with safe defaults
        credit = float(trade_data.get('credit', 0))
        debit = float(trade_data.get('debit_paid', 0))
        contracts = int(trade_data.get('contracts', 1))
        pl = (credit - debit) * contracts * 100
        
        # FIXED: Row order must match the header above
        row = [[
            trade_data.get('ticker'),
            trade_data.get('entry_date'),
            datetime.now().strftime("%Y-%m-%d"), # Exit Date (Today)
            trade_data.get('short_strike'),
            trade_data.get('long_strike'),
            contracts,
            credit,
            debit,
            pl,
            trade_data.get('earnings_date', ''), # Earnings Date
            trade_data.get('notes', '')
        ]]
        
        sheets_service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=f"{SHEET_TITLE}!A1",
            valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS", body={"values": row}
        ).execute()
        return True
    except Exception as e:
        print(f"Log Error: {e}")
        return False

def get_trade_log(service):
    if not service: return []
    try:
        sheets_service = build_sheets_service_from_session()
        spreadsheet_id = _get_spreadsheet_id(service, sheets_service)
        
        res = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=f"{SHEET_TITLE}!A:Z"
        ).execute()
        
        rows = res.get('values', [])
        if len(rows) < 2: return []
        
        headers = rows[0]
        data = []
        for i, r in enumerate(rows[1:]):
            item = {headers[j]: (r[j] if j < len(r) else "") for j in range(len(headers))}
            item['_row_index'] = i + 1 
            data.append(item)
        return data
    except: return []

def delete_log_entry(service, row_index):
    if not service: return False
    try:
        sheets_service = build_sheets_service_from_session()
        spreadsheet_id = _get_spreadsheet_id(service, sheets_service)
        sheet_id = _ensure_sheet_exists(sheets_service, spreadsheet_id)
        
        grid_index = row_index + 1 
        
        req = {
            "deleteDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": grid_index,
                    "endIndex": grid_index + 1
                }
            }
        }
        
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [req]}
        ).execute()
        return True
    except Exception as e:
        print(f"Delete Error: {e}")
        return False
