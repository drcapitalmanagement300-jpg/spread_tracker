import streamlit as st
import json
import os
import io
from datetime import datetime
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# --- CONFIG ---
LOCAL_DB = "trades.json"
JOURNAL_DB = "journal.json"
DATA_FOLDER_ID = None  # We will search for a folder named "SpreadSniper_Data"

# --- AUTH & SESSION ---
def ensure_logged_in():
    if "credentials" not in st.session_state:
        st.warning("Please log in via the Dashboard.")
        st.stop()

def build_drive_service_from_session():
    if "credentials" in st.session_state:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        creds = Credentials(**st.session_state.credentials)
        return build('drive', 'v3', credentials=creds)
    return None

def logout():
    st.session_state.clear()

# --- LARGE DATA HANDLING (The New Magic) ---
def get_or_create_data_folder(service):
    """Finds or creates a folder named 'SpreadSniper_Data' to keep things clean."""
    query = "name='SpreadSniper_Data' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    else:
        file_metadata = {
            'name': 'SpreadSniper_Data',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

def download_large_file_from_drive(service, filename, local_path):
    """Downloads a large file from Drive to Local Path using streaming."""
    folder_id = get_or_create_data_folder(service)
    
    # 1. Search for file
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, size)').execute()
    files = results.get('files', [])
    
    if not files:
        return False # File not in Drive
        
    file_id = files[0]['id']
    file_size = int(files[0].get('size', 0))
    
    # 2. Download Stream
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    
    # UI Progress
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    while done is False:
        status, done = downloader.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            progress_bar.progress(progress / 100)
            progress_text.caption(f"Downloading from Drive: {progress}%")
            
    progress_bar.empty()
    progress_text.empty()
    return True

def upload_large_file_to_drive(service, local_path, filename):
    """Uploads a large local file to Drive for safekeeping."""
    folder_id = get_or_create_data_folder(service)
    
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(local_path, resumable=True)
    
    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    
    # UI Progress
    response = None
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            progress_bar.progress(progress / 100)
            progress_text.caption(f"Backing up to Drive: {progress}%")
            
    progress_bar.empty()
    progress_text.empty()
    return True

# --- EXISTING TRADE PERSISTENCE (Keep this as is) ---
def load_from_drive(service=None):
    if os.path.exists(LOCAL_DB):
        with open(LOCAL_DB, "r") as f: return json.load(f)
    return []

def save_to_drive(service, trades):
    with open(LOCAL_DB, "w") as f: json.dump(trades, f, indent=4)
    return True

def get_trade_log(service=None):
    if os.path.exists(JOURNAL_DB):
        with open(JOURNAL_DB, "r") as f: return json.load(f)
    return []

def log_completed_trade(service, trade_data):
    # (Same logic as before, just kept short here for brevity)
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
    return True

def delete_log_entry(service, index):
    current_log = get_trade_log()
    if 0 <= index < len(current_log):
        current_log.pop(index)
        with open(JOURNAL_DB, "w") as f: json.dump(current_log, f, indent=4)
        return True
    return False
