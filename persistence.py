import streamlit as st
import json
import io
import os
import csv  # Added for CSV logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# --- IMPORT LOCK MANAGER ---
try:
    from google_drive import DriveLockManager
except ImportError:
    DriveLockManager = None

# ----------------------------- Config -----------------------------
DRIVE_FILE_NAME = "credit_spreads.json"
DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]
# token file names
LOCAL_TOKEN_FILE = "google_oauth_token.json"
DRIVE_TOKEN_FILE = "credit_spreads_token.json"

# ----------------------------- OAuth Config Utilities -----------------------------
def _get_oauth_config() -> Dict[str, str]:
    """Load OAuth credentials from Streamlit secrets."""
    if "google_oauth" not in st.secrets:
        st.error("Missing google_oauth in Streamlit secrets.")
        st.stop()

    cfg = st.secrets["google_oauth"]
    for key in ("client_id", "client_secret", "redirect_uri"):
        if key not in cfg:
            st.error(f"google_oauth secret missing required key: {key}")
            st.stop()

    return cfg


def _get_redirect_uri() -> str:
    return st.secrets["google_oauth"]["redirect_uri"]


def get_flow() -> Flow:
    """Create OAuth flow object."""
    cfg = _get_oauth_config()
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [cfg["redirect_uri"]],
            }
        },
        scopes=DRIVE_SCOPES,
    )
    flow.redirect_uri = cfg["redirect_uri"]
    return flow


# ----------------------------- Helper: local token store -----------------------------
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
                data = json.load(f)
                return data
    except Exception:
        return None
    return None


# ----------------------------- Drive file helpers -----------------------------
def _find_file_id(service, filename: str) -> Optional[str]:
    try:
        q = f"name = '{filename}' and trashed = false"
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="files(id,name)"
        ).execute()
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


def _download_file_by_name(service, filename: str) -> Optional[str]:
    file_id = _find_file_id(service, filename)
    if not file_id:
        return None
    return _download_file(service, file_id)


# ----------------------------- OAuth Flow -----------------------------
def exchange_code_for_credentials(code: str) -> Optional[Dict[str, Any]]:
    try:
        flow = get_flow()
        flow.fetch_token(code=code)
        creds = flow.credentials

        creds_dict = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes),
        }

        _save_local_token(creds_dict)
        st.session_state["credentials"] = creds_dict

        try:
            service = build_drive_service_from_session()
            if service:
                _upload_file(service, DRIVE_TOKEN_FILE, json.dumps(creds_dict))
        except Exception:
            pass

        return creds_dict
    except Exception as e:
        st.error(f"OAuth token exchange failed: {e}")
        return None


def logout():
    st.session_state.pop("credentials", None)
    try:
        if os.path.exists(LOCAL_TOKEN_FILE):
            os.remove(LOCAL_TOKEN_FILE)
    except Exception:
        pass
    st.success("Logged out.")
    # UPDATED: Use st.rerun() instead of experimental_rerun()
    st.rerun()


def _credentials_dict_to_oauth(creds_dict: Dict[str, Any]) -> Optional[OAuthCredentials]:
    try:
        if not creds_dict:
            return None
        creds = OAuthCredentials(
            token=creds_dict.get("token"),
            refresh_token=creds_dict.get("refresh_token"),
            token_uri=creds_dict.get("token_uri"),
            client_id=creds_dict.get("client_id"),
            client_secret=creds_dict.get("client_secret"),
            scopes=creds_dict.get("scopes"),
        )
        return creds
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


def build_drive_service_from_session():
    cred_dict = st.session_state.get("credentials")
    if not cred_dict:
        return None

    try:
        creds = _credentials_dict_to_oauth(cred_dict)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            new_dict = _oauth_to_credentials_dict(creds)
            st.session_state["credentials"] = new_dict
            _save_local_token(new_dict)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"Drive service error: {e}")
        return None


def _try_load_credentials() -> Optional[Dict[str, Any]]:
    if st.session_state.get("credentials"):
        return st.session_state.get("credentials")

    local = _load_local_token()
    if local:
        creds = _credentials_dict_to_oauth(local)
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                new = _oauth_to_credentials_dict(creds)
                st.session_state["credentials"] = new
                _save_local_token(new)
                return new
            else:
                st.session_state["credentials"] = local
                return local
        except Exception:
            pass

    return None


def ensure_logged_in():
    """
    Blocks execution until the user is logged in via Google OAuth.
    Uses modern st.query_params to avoid deprecation warnings.
    """
    creds_dict = _try_load_credentials()
    if creds_dict:
        st.session_state["credentials"] = creds_dict
        return

    # UPDATED: Use st.query_params (dict-like) instead of experimental_get_query_params (list-like)
    if "code" in st.query_params and not st.session_state.get("credentials"):
        # The new API returns strings directly, so no need for [0]
        code = st.query_params["code"]
        creds = exchange_code_for_credentials(code)
        if creds:
            st.session_state["credentials"] = creds
            # UPDATED: Clear params using .clear()
            st.query_params.clear()
            # UPDATED: Use st.rerun()
            st.rerun()

    flow = get_flow()
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )
    st.markdown("### Sign in with Google to enable Drive persistence")
    # Using a button that acts as a link is cleaner
    st.link_button("Sign in with Google", auth_url, type="primary")
    st.stop()


# ----------------------------- Drive Helpers (public) -----------------------------

def load_from_drive(service) -> List[Dict[str, Any]]:
    """Load and deserialize trades from Drive."""
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    if not file_id:
        return []

    raw = _download_file(service, file_id)
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # Deserialize date fields
    parsed = []
    for t in data:
        ct = {}
        for k, v in t.items():
            if isinstance(v, str):
                try:
                    if len(v) == 10 and v[4] == '-' and v[7] == '-':  # YYYY-MM-DD
                        ct[k] = date.fromisoformat(v)
                    elif "T" in v and len(v) > 10:  # datetime
                        ct[k] = datetime.fromisoformat(v)
                    else:
                        ct[k] = v
                except Exception:
                    ct[k] = v
            else:
                ct[k] = v
        parsed.append(ct)

    return parsed


def save_to_drive(service, trades: List[Dict[str, Any]]) -> bool:
    """
    Serialize and upload trade list to Drive with Lock protection.
    Includes smart merging to prevent overwriting backend updates.
    """
    
    # 1. Prepare Lock
    file_id = _find_file_id(service, DRIVE_FILE_NAME)
    lock = None
    if DriveLockManager:
        lock = DriveLockManager(service, file_id)
    
    try:
        if lock:
            lock.acquire()
            
        # 2. SMART MERGE STRATEGY
        # Download latest from Drive (Source of Truth for Prices/Deltas)
        latest_drive_trades = load_from_drive(service)
        drive_map = {t['id']: t for t in latest_drive_trades if 'id' in t}
        
        # 'trades' contains the user's intended list (e.g. they deleted one trade).
        # We iterate through the USER'S list, but grab the DATA from the DRIVE list.
        merged_trades = []
        for user_trade in trades:
            t_id = user_trade.get('id')
            
            # If this trade exists on Drive, prefer Drive's cached math/history
            if t_id and t_id in drive_map:
                drive_trade = drive_map[t_id]
                # Merge fresh backend data into user object
                if 'cached' in drive_trade:
                    user_trade['cached'] = drive_trade['cached']
                if 'pnl_history' in drive_trade:
                    user_trade['pnl_history'] = drive_trade['pnl_history']
                if 'last_heartbeat_date' in drive_trade:
                    user_trade['last_heartbeat_date'] = drive_trade['last_heartbeat_date']
            
            merged_trades.append(user_trade)
            
        # 3. Serialize
        serializable = []
        for t in merged_trades:
            ct = {}
            for k, v in t.items():
                if isinstance(v, (date, datetime)):
                    ct[k] = v.isoformat()
                else:
                    ct[k] = v
            serializable.append(ct)

        # 4. Upload
        return _upload_file(service, DRIVE_FILE_NAME, json.dumps(serializable, indent=2))
        
    except Exception as e:
        print(f"Save failed: {e}")
        return False
    finally:
        if lock:
            lock.release()

# ----------------------------- CSV Journaling -----------------------------

def log_trade_to_csv(service, trade_data: Dict[str, Any], debit_paid: float, notes: str) -> bool:
    """
    Logs a closed trade to 'trade_journal.csv' on Drive.
    Calculates Realized PnL and other metrics automatically.
    """
    FILENAME = "trade_journal.csv"
    
    # 1. Calculate Metrics
    try:
        credit = float(trade_data.get("credit", 0))
        # PnL Calculation: (Credit Received - Debit Paid) * 100
        # Assumes 1 contract sizing per entry
        pnl_val = (credit - debit_paid) * 100
        
        entry_date_str = trade_data.get("entry_date", date.today().isoformat())
        # Handle timestamp strings if they include time components
        if isinstance(entry_date
