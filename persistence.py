import streamlit as st
import json
import io
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
# PATCH: Added MediaFileUpload
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload, MediaFileUpload

# --- IMPORT LOCK MANAGER (Optional) ---
try:
    from google_drive import DriveLockManager
except ImportError:
    DriveLockManager = None

# ----------------------------- Config -----------------------------
DRIVE_FILE_NAME = "credit_spreads.json"
SPREADSHEET_NAME = "SpreadSniper_Data"
DATASET_FOLDER_NAME = "SpreadSniper_Datasets" # NEW: Folder for large files

# Scopes
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
LOCAL_TOKEN_FILE = "google_oauth_token.json"

# ----------------------------- OAuth Helpers -----------------------------
def _get_oauth_config() -> Dict[str, str]:
    if "google_oauth" not in st.secrets:
        st.error("Missing google_oauth in Streamlit secrets.")
        st.stop()
    return st.secrets["google_oauth"]

def _get_web_client_config(cfg: Dict[str, str]) -> Dict[str, Any]:
    return {
        "web": {
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uris": [cfg["redirect_uri"]],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def _save_local_token(creds_dict: Dict[str, Any]) -> None:
    try:
        with open(LOCAL_TOKEN_FILE, "w") as f:
            json.dump(creds_dict, f)
    except Exception:
        pass

def _load_local_token() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(LOCAL_TOKEN_FILE):
            with open(LOCAL_TOKEN_FILE, "r") as f:
                return json.load(f)
    except Exception:
        return None

def _credentials_dict_to_oauth(creds_dict: Dict[str, Any]) -> Optional[OAuthCredentials]:
    try:
        if not creds_dict: return None
        return OAuthCredentials(
            token=creds_dict.get("token"),
            refresh_token=creds_dict.get("refresh_token"),
            token_uri=creds_dict.get("token_uri"),
            client_id=creds_dict.get("client_id"),
            client_secret=creds_dict.get("client_secret"),
            scopes=creds_dict.get("scopes"),
        )
    except Exception:
        return None

def _oauth_to_credentials_dict(creds: OAuthCredentials) -> Dict[str, Any]:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else [],
    }

# ----------------------------- Service Builders -----------------------------
def get_creds_from_session():
    cred_dict = st.session_state.get("credentials")
    if not cred_dict:
        cred_dict = _load_local_token()
        if not cred_dict: return None

    try:
        creds = _credentials_dict_to_oauth(cred_dict)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            new_dict = _oauth_to_credentials_dict(creds)
            st.session_state["credentials"] = new_dict
            _save_local_token(new_dict)
            return creds
        return creds
    except Exception:
        return None

def build_drive_service_from_session():
    creds = get_creds_from_session()
    if not creds: return None
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def build_sheets_service_from_session():
    creds = get_creds_from_session()
    if not creds: return None
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

# ----------------------------- Auth Flow -----------------------------
def ensure_logged_in():
    if get_creds_from_session():
        return

    if "code" in st.query_params:
        from google_auth_oauthlib.flow import Flow
        cfg = _get_oauth_config()
        client_config = _get_web_client_config(cfg)
        
        try:
            flow = Flow.from_client_config(
                client_config, scopes=SCOPES, redirect_uri=cfg["redirect_uri"]
            )
            flow.fetch_token(code=st.query_params["code"])
            creds_dict = _oauth_to_credentials_dict(flow.credentials)
            st.session_state["credentials"] = creds_dict
            _save_local_token(creds_dict)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.stop()

    from google_auth_oauthlib.flow import Flow
    cfg = _get_oauth_config()
    client_config = _get_web_client_config(cfg)
    
    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=cfg["redirect_uri"]
    )
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    
    st.markdown(f"### [Sign in with Google]({auth_url})")
    st.stop()

def logout():
    st.session_state.pop("credentials", None)
    if os.path.exists(LOCAL_TOKEN_FILE):
        try: os.remove(LOCAL_TOKEN_FILE)
        except: pass
    st.success("Logged out.")
    st.rerun()

# ----------------------------- Drive Helpers (Internal) -----------------------------
def _find_file_id(service, filename: str) -> Optional[str]:
    try:
        q = f"name = '{filename}' and trashed = false"
        resp = service.files().list(q=q, spaces="drive", fields="files(id,name)").execute()
        files = resp.get("files", [])
        return files[0]["id"] if files else None
    except Exception:
        return None

def _download_file(service, file_id: str) -> Optional[str]:
    try:
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        return buf.read().decode("utf-8")
    except Exception:
        return None

def _upload_file(service, filename: str, content: str) -> bool:
    try:
        file_id = _find_file_id(service, filename)
        data = io.BytesIO(content.encode("utf-8"))
        media = MediaIoBaseUpload(data, mimetype="application/json", resumable=False)
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            service.files().create(body={"name": filename}, media_body=media).execute()
        return True
    except Exception:
        return False

# ----------------------------- LARGE DATASET HANDLING (NEW) -----------------------------
def _get_or_create_data_folder(service):
    """Finds or creates a specific folder for large datasets."""
    q = f"name = '{DATASET_FOLDER_NAME}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    files = service.files().list(q=q, spaces='drive', fields='files(id, name)').execute().get('files', [])
    
    if files:
        return files[0]['id']
    else:
        file_metadata = {
            'name': DATASET_FOLDER_NAME,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')

def download_large_file_from_drive(service, filename, local_path, progress_bar=None):
    """Downloads a large file from Drive directly to disk (Streaming)."""
    try:
        folder_id = _get_or_create_data_folder(service)
        q = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
        files = service.files().list(q=q, spaces='drive', fields='files(id, size)').execute().get('files', [])
        
        if not files:
            return False
            
        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id)
        
        # Stream to disk
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status and progress_bar:
                    progress_bar.progress(int(status.progress() * 100) / 100)
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def upload_large_file_to_drive(service, local_path, filename, progress_bar=None):
    """Uploads a large local file to Drive."""
    try:
        folder_id = _get_or_create_data_folder(service)
        
        # Check if file exists to update
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
    except Exception as e:
        print(f"Upload Error: {e}")
        return False

# ----------------------------- Active Trades (JSON) -----------------------------
def load_from_drive(service) -> List[Dict[str, Any]]:
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    if not file_id: return []
    raw = _download_file(service, file_id)
    if not raw: return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    parsed = []
    for t in data:
        ct = t.copy()
        for k, v in t.items():
            if isinstance(v, str):
                if len(v) == 10 and v[4] == '-' and v[7] == '-':
                    try: ct[k] = date.fromisoformat(v)
                    except: pass
                elif "T" in v and len(v) > 10:
                    try: ct[k] = datetime.fromisoformat(v)
                    except: pass
        parsed.append(ct)
    return parsed

def save_to_drive(service, trades: List[Dict[str, Any]]) -> bool:
    serializable = []
    for t in trades:
        ct = {}
        for k, v in t.items():
            if isinstance(v, (date, datetime)):
                ct[k] = v.isoformat()
            else:
                ct[k] = v
        serializable.append(ct)
    return _upload_file(service, DRIVE_FILE_NAME, json.dumps(serializable, indent=2))

# ----------------------------- LOGGING (Google Sheets) -----------------------------
def _get_or_create_spreadsheet(drive_service, sheets_service):
    q = f"name = '{SPREADSHEET_NAME}' and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
    files = drive_service.files().list(q=q).execute().get('files', [])
    if files: return files[0]['id']
    else:
        spreadsheet = {'properties': {'title': SPREADSHEET_NAME}}
        spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
        return spreadsheet.get('spreadsheetId')

def log_completed_trade(drive_service, trade_data):
    try:
        sheets_service = build_sheets_service_from_session()
        if not sheets_service: return False
        spreadsheet_id = _get_or_create_spreadsheet(drive_service, sheets_service)
        ss_meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_exists = any(s['properties']['title'] == "TradeLog" for s in ss_meta['sheets'])
        if not sheet_exists:
            body = {"requests": [{"addSheet": {"properties": {"title": "TradeLog"}}}]}
            sheets_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
            headers = [["Trade_ID", "Ticker", "Short_Strike", "Long_Strike", "Contracts", "Entry_Date", "Exit_Date", "Credit", "Debit_Paid", "Realized_PL", "Notes"]]
            sheets_service.spreadsheets().values().update(spreadsheetId=spreadsheet_id, range="TradeLog!A1", valueInputOption="RAW", body={"values": headers}).execute()
        
        credit = float(trade_data.get('credit', 0))
        debit = float(trade_data.get('debit_paid', 0))
        contracts = int(trade_data.get('contracts', 1))
        pl = (credit - debit) * contracts * 100
        entry_d = trade_data.get('entry_date', '')
        if isinstance(entry_d, (date, datetime)): entry_d = entry_d.isoformat()
        
        row = [[trade_data.get('id', ''), trade_data.get('ticker', ''), trade_data.get('short_strike', ''), trade_data.get('long_strike', ''), contracts, entry_d, date.today().isoformat(), credit, debit, pl, trade_data.get('notes', '')]]
        sheets_service.spreadsheets().values().append(spreadsheetId=spreadsheet_id, range="TradeLog!A1", valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS", body={"values": row}).execute()
        return True
    except Exception as e:
        print(f"Log Error: {e}")
        return False

def get_trade_log(drive_service):
    try:
        sheets_service = build_sheets_service_from_session()
        if not sheets_service: return []
        q = f"name = '{SPREADSHEET_NAME}'"
        files = drive_service.files().list(q=q).execute().get('files', [])
        if not files: return []
        res = sheets_service.spreadsheets().values().get(spreadsheetId=files[0]['id'], range="TradeLog!A:Z").execute()
        rows = res.get('values', [])
        if len(rows) < 2: return []
        headers = rows[0]
        data = []
        for r in rows[1:]:
            item = {}
            for i, h in enumerate(headers):
                val = r[i] if i < len(r) else ""
                item[h] = val
            data.append(item)
        return data
    except Exception:
        return []

def delete_log_entry(drive_service, row_index):
    try:
        sheets_service = build_sheets_service_from_session()
        if not sheets_service: return False
        q = f"name = '{SPREADSHEET_NAME}'"
        files = drive_service.files().list(q=q).execute().get('files', [])
        if not files: return False
        spreadsheet_id = files[0]['id']
        ss_meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_id = 0
        for s in ss_meta['sheets']:
            if s['properties']['title'] == "TradeLog":
                sheet_id = s['properties']['sheetId']
                break
        sheet_row_index = row_index + 1
        body = {"requests": [{"deleteDimension": {"range": {"sheetId": sheet_id, "dimension": "ROWS", "startIndex": sheet_row_index, "endIndex": sheet_row_index + 1}}}]}
        sheets_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
        return True
    except Exception:
        return False
