# persistence.py

import streamlit as st
import json
import io
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


# ----------------------------- OAuth Flow -----------------------------
def exchange_code_for_credentials(code: str) -> Optional[Dict[str, Any]]:
    """Exchange auth code for token + refresh_token."""
    try:
        flow = get_flow()
        flow.fetch_token(code=code)
        creds = flow.credentials

        # store a serializable form
        return {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes),
        }
    except Exception as e:
        st.error(f"OAuth token exchange failed: {e}")
        return None


def ensure_logged_in():
    """
    Ensures valid Google OAuth session.
    If missing, renders a login link and stops app execution.
    """
    if "credentials" not in st.session_state:
        st.session_state["credentials"] = None

    # Handle Google redirect callback
    q = st.experimental_get_query_params()
    if "code" in q and not st.session_state["credentials"]:
        code = q["code"][0] if isinstance(q["code"], list) else q["code"]
        creds = exchange_code_for_credentials(code)

        if creds:
            st.session_state["credentials"] = creds
            st.experimental_set_query_params()
            st.experimental_rerun()

    # If still no credentials → show Google login link
    if not st.session_state["credentials"]:
        flow = get_flow()
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true",
        )
        st.markdown("### Sign in with Google to enable Drive persistence")
        st.markdown(f"[Click here to sign in with Google]({auth_url})")
        st.stop()


def logout():
    """Clear credentials and force re-login."""
    st.session_state.pop("credentials", None)
    st.success("Logged out.")
    st.experimental_rerun()


# ----------------------------- Drive Helpers -----------------------------
def build_drive_service_from_session():
    """Returns an authenticated Drive service."""
    cred_dict = st.session_state.get("credentials")
    if not cred_dict:
        return None

    try:
        creds = OAuthCredentials(
            token=cred_dict["token"],
            refresh_token=cred_dict["refresh_token"],
            token_uri=cred_dict["token_uri"],
            client_id=cred_dict["client_id"],
            client_secret=cred_dict["client_secret"],
            scopes=cred_dict["scopes"],
        )

        # Refresh if needed
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["credentials"]["token"] = creds.token

        return build("drive", "v3", credentials=creds, cache_discovery=False)

    except Exception as e:
        st.error(f"Drive service error: {e}")
        return None


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


# ----------------------------- Public API -----------------------------
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
