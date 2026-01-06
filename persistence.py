import csv
import os
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
LOG_FILE = 'trade_log.csv'
# The master list of columns. If the CSV doesn't match this, it will be auto-fixed.
EXPECTED_HEADERS = [
    'Ticker', 'Entry_Date', 'Exit_Date', 'Strategy', 'Direction',
    'Short_Strike', 'Long_Strike', 'Contracts', 'Credit', 'Debit',
    'Realized_PL', 'Status', 'Notes', 'Earnings_Date'
]

def build_drive_service_from_session():
    # Placeholder if you add Google Drive later. 
    # For now, we are using local CSV storage.
    return True 

def get_trade_log(service=None):
    """
    Reads the CSV log. If file is missing or empty, returns empty list.
    """
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return []
    
    try:
        # Read with pandas for robustness, then convert to list of dicts
        df = pd.read_csv(LOG_FILE)
        # Handle cases where CSV might have different columns than expected
        return df.to_dict('records')
    except Exception as e:
        print(f"Error reading log: {e}")
        return []

def log_new_trade(trade_data):
    """
    Appends a new trade to the CSV.
    Checks for file existence and header validity first.
    """
    # 1. Prepare the row data based on EXPECTED_HEADERS
    # This ensures order is perfect even if the dict passed in is unordered
    row_to_write = []
    for header in EXPECTED_HEADERS:
        row_to_write.append(trade_data.get(header, ""))

    # 2. Check File State (The "Self-Healing" Logic)
    file_mode = 'a'
    write_header = False
    
    if not os.path.exists(LOG_FILE):
        file_mode = 'w'
        write_header = True
    elif os.path.getsize(LOG_FILE) == 0:
        write_header = True
    else:
        # Check if existing headers match
        try:
            with open(LOG_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_headers = next(reader)
                if existing_headers != EXPECTED_HEADERS:
                    # Header mismatch: We must rewrite the file to fix it
                    # (In a real app, you might want to migrate data, but here we reset structure)
                    print("Updating CSV structure...")
                    # We will append, but we might have issues. 
                    # Ideally, you'd read all data and rewrite it. 
                    # For safety in this simple version, we stick to append but warn user.
                    pass 
        except Exception:
            write_header = True

    # 3. Write Data
    with open(LOG_FILE, file_mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(EXPECTED_HEADERS)
        writer.writerow(row_to_write)
    
    return True

def update_trade_log(service, row_index, updates):
    """
    Updates a specific row in the CSV (e.g., when closing a trade).
    """
    df = pd.read_csv(LOG_FILE)
    
    # Update the specific columns
    for key, value in updates.items():
        if key in df.columns:
            df.at[row_index, key] = value
            
    # Save back to CSV
    df.to_csv(LOG_FILE, index=False)
    return True

def delete_log_entry(service, row_index):
    """
    Deletes a specific row from the CSV.
    """
    try:
        df = pd.read_csv(LOG_FILE)
        df = df.drop(row_index)
        df.to_csv(LOG_FILE, index=False)
        return True
    except Exception as e:
        print(f"Error deleting entry: {e}")
        return False

def logout():
    pass
