import json
import os
import time
import logging
from datetime import datetime, timedelta, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
import io

# Setup simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# -------------------- Lock Manager Class --------------------
class DriveLockManager:
    """
    Handles creating/checking a 'lock' file in Google Drive to prevent
    concurrent writes between the Streamlit App and the GitHub Action.
    """
    def __init__(self, service, file_id, lock_filename="trade_system.lock", timeout_minutes=5):
        self.service = service
        self.data_file_id = file_id
        self.lock_filename = lock_filename
        self.timeout_minutes = timeout_minutes
        self.parent_id = self._get_parent_folder_id()

    def _get_parent_folder_id(self):
        """Finds the folder containing the trades.json file."""
        try:
            # If file_id is None, we can't lock. This happens on first run if file doesn't exist.
            if not self.data_file_id:
                return None
                
            file = self.service.files().get(
                fileId=self.data_file_id, fields='parents'
            ).execute()
            parents = file.get('parents')
            if parents:
                return parents[0]
            # If no parent (root), returning 'root' is usually safe in Drive API
            return 'root' 
        except Exception as e:
            logger.error(f"Could not determine parent folder: {e}")
            return None

    def _get_lock_file_id(self):
        """Checks if the lock file exists and returns its metadata."""
        if not self.parent_id: return None
        
        query = (f"name = '{self.lock_filename}' and "
                 f"'{self.parent_id}' in parents and "
                 f"trashed = false")
        try:
            results = self.service.files().list(
                q=query, fields="files(id, createdTime)"
            ).execute()
            files = results.get('files', [])
            return files[0] if files else None
        except Exception:
            return None

    def acquire(self):
        """Attempts to acquire the lock. Waits if locked."""
        if not self.parent_id:
            logger.warning("No parent folder found (new file?), skipping lock.")
            return True

        logger.info("Attempting to acquire system lock...")
        retries = 0
        max_retries = 20 # Wait up to ~100 seconds
        
        while retries < max_retries:
            lock_file = self._get_lock_file_id()
            
            if lock_file:
                # Lock exists. Check if it's stale.
                created_str = lock_file['createdTime'] 
                # Basic ISO parsing (Drive API returns UTC 'Z')
                try:
                    # Python <3.11 handle 'Z' workaround
                    if created_str.endswith('Z'):
                        created_str = created_str[:-1] + '+00:00'
                    created_dt = datetime.fromisoformat(created_str)
                    
                    age = datetime.now(timezone.utc) - created_dt
                    
                    if age > timedelta(minutes=self.timeout_minutes):
                        logger.warning(f"Found stale lock ({age} old). Force removing.")
                        self.release() # Force delete
                    else:
                        logger.info("System locked by another process. Waiting...")
                        time.sleep(5)
                        retries += 1
                        continue
                except Exception as e:
                    # If date parsing fails, ignore and wait
                    logger.warning(f"Error checking lock age: {e}")
                    time.sleep(5)
                    retries += 1
                    continue

            # Create the lock file
            file_metadata = {
                'name': self.lock_filename,
                'parents': [self.parent_id],
                'mimeType': 'application/vnd.google-apps.folder' # Folder type is faster to create
            }
            try:
                self.service.files().create(body=file_metadata).execute()
                logger.info("Lock acquired.")
                return True
            except HttpError:
                # Race condition: someone else made it milliseconds ago
                logger.warning("Lock contention detected.")
                time.sleep(2)
                retries += 1
        
        raise TimeoutError("Could not acquire lock after multiple attempts.")

    def release(self):
        """Deletes the lock file."""
        try:
            lock_file = self._get_lock_file_id()
            if lock_file:
                self.service.files().delete(fileId=lock_file['id']).execute()
                logger.info("Lock released.")
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")


# -------------------- Existing Logic --------------------

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        "spread-monitor-service.json",  # <-- your downloaded key
        scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def find_file(service, filename):
    results = service.files().list(
        q=f"name='{filename}' and trashed=false",
        spaces="drive",
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def save_trades_to_drive(trades, filename="trades.json"):
    service = get_drive_service()
    file_id = find_file(service, filename)
    
    # Initialize Lock
    lock = DriveLockManager(service, file_id)
    
    try:
        # Acquire Lock before writing
        lock.acquire()
        
        # NOTE: Ideally, here you would read the file *again* to merge changes, 
        # but for now we secure the write operation.
        
        data_str = json.dumps(trades, indent=2)
        fh = io.BytesIO(data_str.encode("utf-8"))
        media = MediaIoBaseUpload(fh, mimetype="application/json")

        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            file_metadata = {"name": filename}
            service.files().create(body=file_metadata, media_body=media).execute()
            
    except Exception as e:
        logger.error(f"Failed to save to Drive: {e}")
        raise
    finally:
        # Always release the lock
        lock.release()

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
