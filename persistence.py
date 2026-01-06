import streamlit as st
import json
import io
import os
import csv
from datetime import datetime, date

# --- GOOGLE AUTH IMPORTS ---
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
import google.auth.exceptions
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from google_auth_oauthlib.flow import Flow

# --- CONFIG ---
TRADES_FILE_NAME = "open_trades.json"
LOG_FILE_NAME = "trade_log.csv"

# THE MASTER HEADER LIST
# If the file on Drive doesn't match this, it will be WIPED and RESET.
EXPECTED_HEADERS = [
    'Ticker', 'Entry_Date', 'Exit_Date', 'Strategy', 'Direction',
    'Short_Strike', 'Long_Strike', 'Contracts', 'Credit', 'Debit',
    'Realized_PL', 'Status', 'Notes', 'Earnings_Date'
]

SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
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
    creds_dict = st.session_state.get("credentials")
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
    st.warning("Please sign in to access your Google Drive.")
    st.markdown(f"[**Sign in with Google**]({auth_url})", unsafe_allow_html=True)
    st.stop()

def logout():
    st.session_state.clear()
    if os.path.exists(LOCAL_TOKEN_FILE):
        os.remove(LOCAL_TOKEN_FILE)
    st.rerun()

def build_drive_service_from_session():
    creds = get_creds_from_session()
    if creds: return build('drive', 'v3', credentials=creds)
    return None

# ----------------------------- DRIVE HELPERS -----------------------------
def _find_file_id(service, filename):
    try:
        q = f"name = '{filename}' and trashed = false"
        results = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
        files = results.get('files', [])
        return files[0]['id'] if files else None
    except: return None

# ----------------------------- OPEN TRADES (JSON) -----------------------------
def load_from_drive(service):
    if not service: return []
    file_id = _find_file_id(service, TRADES_FILE_NAME)
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
    file_id = _find_file_id(service, TRADES_FILE_NAME)
    
    try:
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            service.files().create(body={'name': TRADES_FILE_NAME}, media_body=media).execute()
        return True
    except: return False

# ----------------------------- CLOSED LOG (CSV) -----------------------------

def _check_and_fix_csv_headers(service, file_id):
    """
    Downloads the first line of the CSV.
    If headers don't match EXPECTED_HEADERS, it deletes the file so it can be recreated.
    Returns True if file is good, False if it was deleted (needs recreation).
    """
    try:
        # Download just the beginning (we can't easily partial download with this lib, 
        # so we download content to memory)
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        content = fh.read().decode('utf-8')
        
        # Check if empty or bad headers
        if not content.strip():
            # Empty file
            service.files().delete(fileId=file_id).execute()
            return False
            
        first_line = content.splitlines()[0]
        expected_line = ",".join(EXPECTED_HEADERS)
        
        # Simple string comparison (robust enough for exact match requirement)
        # We strip carriage returns just in case
        if first_line.strip() != expected_line.strip():
            print("CSV Header mismatch. Wiping file...")
            service.files().delete(fileId=file_id).execute()
            return False
            
        return True
    except Exception as e:
        print(f"Error checking CSV: {e}")
        return True # Assume safe to prevent loops, or could be False

def log_completed_trade(service, trade_data):
    """
    Appends a closed trade to the CSV on Drive.
    """
    if not service: return False
    
    # 1. Calculate P&L
    try:
        credit = float(trade_data.get('credit', 0))
        debit = float(trade_data.get('debit_paid', 0))
        contracts = int(trade_data.get('contracts', 1))
        pnl = (credit - debit) * contracts * 100
    except:
        pnl = 0

    # 2. Prepare Row Dict
    row_dict = {
        'Ticker': trade_data.get('ticker'),
        'Entry_Date': trade_data.get('entry_date'),
        'Exit_Date': trade_data.get('exit_date'),
        'Strategy': trade_data.get('strategy', 'Credit Spread'),
        'Direction': trade_data.get('direction', 'Neutral'),
        'Short_Strike': trade_data.get('short_strike'),
        'Long_Strike': trade_data.get('long_strike'),
        'Contracts': contracts,
        'Credit': credit,
        'Debit': debit,
        'Realized_PL': pnl,
        'Status': 'Closed',
        'Notes': trade_data.get('notes', ''),
        'Earnings_Date': trade_data.get('earnings_date', '')
    }

    # 3. Handle Drive File
    file_id = _find_file_id(service, LOG_FILE_NAME)
    
    # Self-Healing: Check structure if it exists
    if file_id:
        is_good = _check_and_fix_csv_headers(service, file_id)
        if not is_good:
            file_id = None # Forces recreation below

    # 4. Prepare Content
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=EXPECTED_HEADERS)
    
    if not file_id:
        # New File: Write Header + Row
        writer.writeheader()
        writer.writerow(row_dict)
        media = MediaIoBaseUpload(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv')
        service.files().create(body={'name': LOG_FILE_NAME}, media_body=media).execute()
    else:
        # Existing File: Append Row
        # Note: Drive API doesn't support "append" directly. We must download, append, upload.
        # Efficient approach for small logs: Download, Append, Update.
        
        # Download existing
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        existing_content = fh.getvalue().decode('utf-8')
        
        # Append new line
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=EXPECTED_HEADERS)
        writer.writerow(row_dict)
        new_row_csv = output.getvalue()
        
        final_content = existing_content
        if not final_content.endswith('\n'):
            final_content += '\n'
        final_content += new_row_csv
        
        media = MediaIoBaseUpload(io.BytesIO(final_content.encode('utf-8')), mimetype='text/csv')
        service.files().update(fileId=file_id, media_body=media).execute()

    return True

def get_trade_log(service):
    """
    Reads the CSV from Drive and returns a list of dicts.
    """
    if not service: return []
    file_id = _find_file_id(service, LOG_FILE_NAME)
    if not file_id: return []
    
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            
        fh.seek(0)
        content = fh.read().decode('utf-8')
        
        return list(csv.DictReader(io.StringIO(content)))
    except: return []

def delete_log_entry(service, row_index):
    """
    Deletes a row by index.
    """
    if not service: return False
    file_id = _find_file_id(service, LOG_FILE_NAME)
    if not file_id: return False
    
    try:
        # Download
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: status, done = downloader.next_chunk()
        
        content = fh.getvalue().decode('utf-8')
        rows = list(csv.DictReader(io.StringIO(content)))
        
        if 0 <= row_index < len(rows):
            del rows[row_index]
            
            # Rewrite
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=EXPECTED_HEADERS)
            writer.writeheader()
            writer.writerows(rows)
            
            media = MediaIoBaseUpload(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv')
            service.files().update(fileId=file_id, media_body=media).execute()
            return True
        return False
    except: return False
