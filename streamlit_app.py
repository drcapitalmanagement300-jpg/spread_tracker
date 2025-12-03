import streamlit as st
from datetime import datetime
import json
import io
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

st.set_page_config(page_title="Credit Spread Monitor", layout="wide")
st.title("Options Spread Monitor")

# ==========================================
# 1. OAuth Login (Streamlit built-in)
# ==========================================

with st.sidebar:
    st.subheader("Google Login")
    login_info = st.login("google", use_cookie=True)

if not login_info.authenticated:
    st.warning("Please sign in with Google to load/save your spread data.")
    st.stop()

creds = Credentials(
    token=login_info.token["access_token"],
    refresh_token=login_info.token.get("refresh_token"),
    token_uri="https://oauth2.googleapis.com/token",
    client_id=login_info.client_info["client_id"],
    client_secret=login_info.client_info["client_secret"]
)

drive_service = build("drive", "v3", credentials=creds)

# ==========================================
# 2. Google Drive Helpers
# ==========================================

FOLDER_NAME = "spread-monitor-data"
FILE_NAME = "spreads.json"

def get_or_create_folder():
    """Find or create the Drive folder that stores the JSON."""
    query = f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder'"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    file_metadata = {
        "name": FOLDER_NAME,
        "mimeType": "application/vnd.google-apps.folder"
    }
    created = drive_service.files().create(body=file_metadata, fields="id").execute()
    return created["id"]


def get_file_id(folder_id):
    """Find the JSON file inside the folder."""
    query = f"'{folder_id}' in parents and name='{FILE_NAME}'"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def load_json_from_drive():
    """Download JSON from Drive or return empty list."""
    folder_id = get_or_create_folder()
    file_id = get_file_id(folder_id)

    if not file_id:
        return []

    request = drive_service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    return json.loads(file_stream.read().decode("utf-8"))


def save_json_to_drive(data):
    """Upload JSON to Drive (create if needed)."""
    folder_id = get_or_create_folder()
    file_id = get_file_id(folder_id)

    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(json_bytes), mimetype="application/json")

    metadata = {"name": FILE_NAME, "parents": [folder_id]}

    if file_id:
        drive_service.files().update(
            fileId=file_id,
            body=metadata,
            media_body=media
        ).execute()
    else:
        drive_service.files().create(
            body=metadata,
            media_body=media,
            fields="id"
        ).execute()

# ==========================================
# 3. Load Data
# ==========================================

if "spreads" not in st.session_state:
    try:
        st.session_state.spreads = load_json_from_drive()
        st.success("Loaded saved data from Drive.")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.session_state.spreads = []

spreads = st.session_state.spreads

# ==========================================
# 4. UI — Add new spread
# ==========================================

st.subheader("Add New Spread")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker").upper()

with col2:
    entry_date = st.date_input("Entry Date", datetime.now())

with col3:
    credit = st.number_input("Credit Received", min_value=0.0, format="%.2f")

if st.button("Add Spread"):
    new_item = {
        "ticker": ticker,
        "entry_date": str(entry_date),
        "credit": credit,
        "created": datetime.now().isoformat()
    }
    spreads.append(new_item)
    save_json_to_drive(spreads)
    st.success("Spread added & saved to Drive.")

# ==========================================
# 5. Display Table
# ==========================================

st.subheader("Current Spreads")
if spreads:
    st.table(spreads)
else:
    st.info("No spreads yet. Add one above.")

# ==========================================
# 6. Save Button
# ==========================================

if st.button("Force Save to Drive"):
    save_json_to_drive(spreads)
    st.success("Saved to Drive.")
