# app.py
import streamlit as st
from datetime import date, datetime
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import altair as alt
import json
import io
from typing import List, Dict, Any, Optional

# OAuth / Google API imports
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

st.set_page_config(page_title="Put Credit Spread Monitor", layout="wide")
st.title("Put Credit Spread Monitor")

# ----------------------------- Config / constants -----------------------------
DRIVE_FILE_NAME = "credit_spreads.json"
DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

# ----------------------------- OAuth helpers -----------------------------
def _get_redirect_uri() -> str:
    try:
        return st.secrets["google_oauth"]["redirect_uri"]
    except Exception:
        return None

def get_flow() -> Flow:
    client_id = st.secrets["google_oauth"]["client_id"]
    client_secret = st.secrets["google_oauth"]["client_secret"]
    redirect_uri = _get_redirect_uri()
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=DRIVE_SCOPES,
    )
    flow.redirect_uri = redirect_uri
    return flow

def ensure_logged_in():
    if "credentials" not in st.session_state:
        st.session_state["credentials"] = None

    # Exchange the code (callback from Google)
    if "code" in st.query_params and st.session_state.get("credentials") is None:
        try:
            flow = get_flow()
            code = st.query_params["code"]
            if isinstance(code, list):
                code = code[0]

            flow.fetch_token(code=code)
            creds = flow.credentials

            st.session_state["credentials"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }

            st.rerun()

        except Exception as e:
            st.error(f"OAuth token exchange failed: {e}")
            st.stop()

    # Not logged in → show login button
    if st.session_state.get("credentials") is None:
        flow = get_flow()
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
            include_granted_scopes="true"
        )
        st.markdown("### Sign in with Google to enable Drive persistence")
        st.markdown(f"[Click here to sign in with Google]({auth_url})")
        st.info("You must sign in to save/read trades to/from Google Drive.")
        st.stop()

def build_drive_service_from_session() -> Optional[object]:
    creds_dict = st.session_state.get("credentials")
    if not creds_dict:
        return None
    try:
        creds = Credentials(
            token=creds_dict.get("token"),
            refresh_token=creds_dict.get("refresh_token"),
            token_uri=creds_dict.get("token_uri"),
            client_id=creds_dict.get("client_id"),
            client_secret=creds_dict.get("client_secret"),
            scopes=creds_dict.get("scopes"),
        )

        # Refresh token if expired
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                st.session_state["credentials"]["token"] = creds.token
            except Exception as e:
                st.warning(f"Could not refresh credentials: {e}")

        return build("drive", "v3", credentials=creds, cache_discovery=False)

    except Exception as e:
        st.error(f"Failed to build Drive service: {e}")
        return None

def logout():
    if "credentials" in st.session_state:
        st.session_state.pop("credentials", None)
    st.success("Logged out. Reload the page to sign in again.")
    st.rerun()

# ----------------------------- Drive helpers -----------------------------
def _get_folder_id() -> Optional[str]:
    try:
        return st.secrets.get("DRIVE_FOLDER_ID", None)
    except Exception:
        return None

def _find_file_id(service, filename: str) -> Optional[str]:
    if service is None:
        return None
    try:
        folder_id = _get_folder_id()
        safe_name = filename.replace("'", "\\'")
        if folder_id:
            query = f"'{folder_id}' in parents and name = '{safe_name}' and trashed = false"
        else:
            query = f"name = '{safe_name}' and trashed = false"
        response = service.files().list(
            q=query, spaces="drive", fields="files(id,name)"
        ).execute()
        files = response.get("files", [])
        return files[0]["id"] if files else None
    except Exception as e:
        st.error(f"Drive find file error: {e}")
        return None

def _download_file(service, file_id: str) -> Optional[str]:
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read().decode("utf-8")
    except Exception as e:
        st.error(f"Drive download error: {e}")
        return None

def _upload_file(service, filename: str, content_str: str) -> bool:
    try:
        folder_id = _get_folder_id()
        file_id = _find_file_id(service, filename)
        fh = io.BytesIO(content_str.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="application/json", resumable=False)

        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            body = {"name": filename}
            if folder_id:
                body["parents"] = [folder_id]
            service.files().create(body=body, media_body=media).execute()
        return True
    except Exception as e:
        st.error(f"Drive upload error: {e}")
        return False

def save_to_drive(service, trades: List[Dict[str, Any]]) -> bool:
    if service is None:
        st.warning("Drive service not available; cannot save.")
        return False
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
    if service is None:
        return []
    try:
        file_id = _find_file_id(service, DRIVE_FILE_NAME)
        if not file_id:
            return []
        raw = _download_file(service, file_id)
        if not raw:
            return []
        loaded = json.loads(raw)
        out = []
        for t in loaded:
            nt = {}
            for k, v in t.items():
                if isinstance(v, str) and k in ("expiration", "entry_date", "created_at"):
                    try:
                        parsed = datetime.fromisoformat(v)
                        if k == "created_at":
                            nt[k] = v
                        else:
                            nt[k] = parsed.date()
                    except Exception:
                        nt[k] = v
                else:
                    nt[k] = v
            out.append(nt)
        return out
    except Exception as e:
        st.error(f"Drive load error: {e}")
        return []

# ----------------------------- Original app helpers -----------------------------
def init_state(drive_service):
    if "trades" not in st.session_state:
        try:
            st.session_state.trades = load_from_drive(drive_service) or []
        except Exception:
            st.session_state.trades = []

def days_to_expiry(exp: date) -> int:
    return max((exp - date.today()).days, 0)

def compute_derived(t: Dict[str, Any]) -> Dict[str, Any]:
    short = float(t["short_strike"])
    long = float(t["long_strike"])
    credit = float(t.get("credit", 0) or 0)
    width = abs(long - short)
    return {
        "width": width,
        "max_gain": credit,
        "max_loss": max(width - credit, 0),
        "breakeven": short + credit,
        "dte": days_to_expiry(t["expiration"])
    }

def format_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"

@st.cache_data(ttl=60)
def get_price(ticker: str):
    try:
        return float(yf.Ticker(ticker).fast_info["last_price"])
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiration: str):
    try:
        opt_chain = yf.Ticker(ticker).option_chain(expiration)
        return opt_chain.calls, opt_chain.puts
    except Exception:
        return None, None

def bsm_delta(option_type, S, K, T, r, sigma):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == "call":
        return norm
