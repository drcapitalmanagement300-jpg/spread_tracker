# persistence.py (upgraded persistent OAuth token storage + Drive backup)
import streamlit as st
import json
import io
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

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
        # best-effort; don't break app
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


# ----------------------------- Drive file helpers (existing) -----------------------------
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
    """Exchange auth code for token + refresh_token."""
    try:
        flow = get_flow()
        flow.fetch_token(code=code)
        creds = flow.credentials

        # store a serializable form
        creds_dict = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes),
        }

        # persist locally immediately (best-effort)
        _save_local_token(creds_dict)
        # also set into session_state for immediate use
        st.session_state["credentials"] = creds_dict

        # Attempt to persist to Drive using a Drive service built from these credentials
        try:
            service = build_drive_service_from_session()
            if service:
                _upload_file(service, DRIVE_TOKEN_FILE, json.dumps(creds_dict))
        except Exception:
            # ignore drive-save errors
            pass

        return creds_dict
    except Exception as e:
        st.error(f"OAuth token exchange failed: {e}")
        return None


def logout():
    """Clear credentials and force re-login."""
    st.session_state.pop("credentials", None)
    # also remove local token file if present
    try:
        if os.path.exists(LOCAL_TOKEN_FILE):
            os.remove(LOCAL_TOKEN_FILE)
    except Exception:
        pass
    st.success("Logged out.")
    st.experimental_rerun()


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
    """Returns an authenticated Drive service using st.session_state credentials."""
    cred_dict = st.session_state.get("credentials")
    if not cred_dict:
        return None

    try:
        creds = _credentials_dict_to_oauth(cred_dict)
        # Refresh if needed
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # update session_state and local token copy
            new_dict = _oauth_to_credentials_dict(creds)
            st.session_state["credentials"] = new_dict
            _save_local_token(new_dict)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as e:
        st.error(f"Drive service error: {e}")
        return None


def _try_load_credentials_from_drive_service(service) -> Optional[Dict[str, Any]]:
    try:
        raw = _download_file_by_name(service, DRIVE_TOKEN_FILE)
        if not raw:
            return None
        data = json.loads(raw)
        return data
    except Exception:
        return None


def _try_load_credentials() -> Optional[Dict[str, Any]]:
    """
    Try to load credentials from (1) session_state (2) local file (3) Drive token file (using temporary flow if needed).
    Returns a credentials dict or None.
    """
    # 1) session_state
    if st.session_state.get("credentials"):
        return st.session_state.get("credentials")

    # 2) local file
    local = _load_local_token()
    if local:
        # attempt to refresh if needed
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
            # fall through to try drive
            pass

    # 3) Try to load token stored on Drive (if we can authenticate via ephemeral flow)
    # This is tricky because to access Drive we need creds; but if we have NONE we can try a flow to fetch code.
    # Instead, try using any available local client secret flow to prompt user - but we prefer local file already.
    # We'll attempt to use an interactive flow only if we have a 'code' query param (handled elsewhere).
    return None


def ensure_logged_in():
    """
    Ensures valid Google OAuth session.
    If missing, renders a login link and stops app execution.
    This function now tries:
      - session_state
      - local token file
      - (attempts silent refresh)
      - otherwise present OAuth consent URL (prompt=consent, access_type=offline)
    """
    # Attempt to load any existing credentials
    creds_dict = _try_load_credentials()
    if creds_dict:
        # ensure it's set in session_state
        st.session_state["credentials"] = creds_dict
        return

    # Handle Google redirect callback (code param)
    q = st.experimental_get_query_params()
    if "code" in q and not st.session_state.get("credentials"):
        code = q["code"][0] if isinstance(q["code"], list) else q["code"]
        creds = exchange_code_for_credentials(code)
        if creds:
            st.session_state["credentials"] = creds
            st.experimental_set_query_params()
            st.experimental_rerun()

    # If still not authenticated → show Google login link
    flow = get_flow()
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )
    st.markdown("### Sign in with Google to enable Drive persistence")
    st.markdown(f"[Click here to sign in with Google]({auth_url})")
    st.stop()


# ----------------------------- Drive Helpers (public) -----------------------------
def save_to_drive(service, trades: List[Dict[str, Any]]) -> bool:
    """Serialize and upload trade list to Drive."""
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
                # Try parsing ISO dates
                try:
                    if len(v) == 10:  # YYYY-MM-DD
                        ct[k] = date.fromisoformat(v)
                    elif "T" in v:  # datetime
                        ct[k] = datetime.fromisoformat(v)
                    else:
                        ct[k] = v
                except Exception:
                    ct[k] = v
            else:
                ct[k] = v
        parsed.append(ct)

    return parsed
