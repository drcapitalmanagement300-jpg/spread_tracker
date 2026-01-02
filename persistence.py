import streamlit as st
import json
import os
import io
import google_auth_oauthlib.flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from datetime import datetime

# --- CONSTANTS ---
LOCAL_DB = "trades.json"
JOURNAL_DB = "journal.json"
CLIENT_SECRETS_FILE = "client_secret.json" 
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# --- AUTHENTICATION FLOW ---
def auth_flow():
    """Runs the Google OAuth flow and reloads the page on success."""
    try:
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS_FILE, SCOPES)
        
        # Run local server for auth
        creds = flow.run_local_server(port=0)
        
        # Store credentials in session state
        st.session_state.credentials = {
            'token': creds.token,
            'refresh_token': creds.refresh_token,
            'token_uri': creds.token_uri,
            'client_id': creds.client_id,
            'client_secret': creds.client_secret,
            'scopes': creds.scopes
        }
        st.rerun()
    except Exception as e:
        st.error(f"Authentication Failed: {e}")

def ensure_logged_in():
    """
    BLOCKING CALL: Checks if user is logged in.
    If NOT logged in, it renders the Sign-In button and STOPS script execution.
    """
    if "credentials" not in st.session_state:
        # Draw the Login Screen
        st.markdown("### Security Check")
        st.warning("Please sign in to access your Trading Dashboard.")
        
        if os.path.exists(CLIENT_SECRETS_FILE):
            if st.button("Sign in with Google", type="primary", use_container_width=True):
                auth_flow()
        else:
            st.error(f"Configuration Error: '{CLIENT_SECRETS_FILE}' not found. Cannot authenticate.")
        
        # CRITICAL: Stop the script here so the Dashboard doesn't try to load
        st.stop()

def logout():
    st.session_state.clear()

def build_drive_service_from_session():
    """Builds the Drive Service from session credentials."""
    if "credentials" in st.session_state:
        try:
            creds = Credentials(**st.session_state.credentials)
            return build('drive', 'v3', credentials=creds)
        except:
            return None
    return None

# --- LARGE FILE HANDLING (DRIVE) ---
def get_or_create_data_folder(service):
    query = "name='SpreadSniper_Data' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    if files: return files[0]['id']
    else:
        file_metadata = {'name': 'SpreadSniper_Data', 'mimeType': 'application/vnd.google-apps.folder'}
        file = service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

def download_large_file_from_drive(service, filename, local_path):
    if not service: return False
    try:
        folder_id = get_or_create_data_folder(service)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, size)').execute()
        files = results.get('files', [])
        if not files: return False
        
        request = service.files().get_media(fileId=files[0]['id'])
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False: status, done = downloader.next_chunk()
        return True
    except: return False

def upload_large_file_to_drive(service, local_path, filename):
    if not service: return False
    try:
        folder_id = get_or_create_data_folder(service)
        file_metadata = {'name': filename, 'parents': [folder_id]}
        media = MediaFileUpload(local_path, resumable=True)
        service.files().create(body=file_metadata, media_body=media).execute()
        return True
    except: return False

# --- TRADE PERSISTENCE ---
def load_from_drive(service=None):
    # Try Local First (Speed)
    if os.path.exists(LOCAL_DB):
        with open(LOCAL_DB, "r") as f: return json.load(f)
    return []

def save_to_drive(service, trades):
    # Save Local
    with open(LOCAL_DB, "w") as f: json.dump(trades, f, indent=4)
    # Sync Drive
    if service:
        try: upload_large_file_to_drive(service, LOCAL_DB, "trades_backup.json")
        except: pass
    return True

def get_trade_log(service=None):
    if os.path.exists(JOURNAL_DB):
        with open(JOURNAL_DB, "r") as f: return json.load(f)
    return []

def log_completed_trade(service, trade_data):
    credit = trade_data.get('credit', 0)
    debit = trade_data.get('debit_paid', 0)
    contracts = trade_data.get('contracts', 1)
    realized_pl = (credit - debit) * 100 * contracts
    
    log_entry = {
        "Ticker": trade_data.get('ticker'),
        "Entry_Date": trade_data.get('entry_date'),
        "Exit_Date": datetime.now().strftime("%Y-%m-%d"),
        "Short_Strike": trade_data.get('short_strike'),
        "Long_Strike": trade_data.get('long_strike'),
        "Credit": credit,
        "Debit_Paid": debit,
        "Contracts": contracts,
        "Realized_PL": realized_pl,
        "Notes": trade_data.get('notes', ''),
        "Duration": (datetime.now().date() - datetime.strptime(trade_data.get('entry_date'), "%Y-%m-%d").date()).days
    }
    
    current_log = get_trade_log()
    current_log.append(log_entry)
    
    with open(JOURNAL_DB, "w") as f: json.dump(current_log, f, indent=4)
    
    if service:
        try: upload_large_file_to_drive(service, JOURNAL_DB, "journal_backup.json")
        except: pass
        
    return True

def delete_log_entry(service, index):
    current_log = get_trade_log()
    if 0 <= index < len(current_log):
        current_log.pop(index)
        with open(JOURNAL_DB, "w") as f: json.dump(current_log, f, indent=4)
        if service:
            try: upload_large_file_to_drive(service, JOURNAL_DB, "journal_backup.json")
            except: pass
        return True
    return False
