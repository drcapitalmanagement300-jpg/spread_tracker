import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import io

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        "spread-monitor-service.json",  # <-- your downloaded key
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def find_file(service, filename):
    results = service.files().list(
        q=f"name='{filename}'",
        spaces="drive",
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def save_trades_to_drive(trades, filename="trades.json"):
    service = get_drive_service()
    file_id = find_file(service, filename)
    data_str = json.dumps(trades, indent=2)
    fh = io.BytesIO(data_str.encode("utf-8"))

    media = MediaIoBaseUpload(fh, mimetype="application/json")

    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        file_metadata = {"name": filename}
        service.files().create(body=file_metadata, media_body=media).execute()

def load_trades_from_drive(filename="trades.json"):
    service = get_drive_service()
    file_id = find_file(service, filename)
    if not file_id:
        return None

    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.seek(0)
    return json.loads(fh.read().decode("utf-8"))
