# gdrive_match.py (Fixed)

import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import cv2
import numpy as np
from deepface import DeepFace

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    """Creates a service object for the Google Drive API."""
    # Load credentials from environment variable
    cred_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not cred_json:
        raise ValueError("Environment variable 'GOOGLE_APPLICATION_CREDENTIALS_JSON' not set.")
    
    creds_info = json.loads(cred_json)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def get_all_gdrive_folder_ids():
    """Fetches all Google Drive folder IDs from the 'photo_sessions' collection in Firestore."""
    from firebase_admin import firestore
    db = firestore.client()
    folder_ids = []
    try:
        sessions_stream = db.collection('photo_sessions').stream()
        for doc in sessions_stream:
            data = doc.to_dict()
            drive_link = data.get('driveLink', '')
            if 'folders/' in drive_link:
                folder_id = drive_link.split('folders/')[1].split('?')[0]
                folder_ids.append(folder_id)
    except Exception as e:
        print(f"[ERROR] Could not fetch folder IDs from Firestore: {e}")
    return folder_ids

def list_photo_links_from_folder(service, folder_id):
    """Gets a list of photo metadata from a specific Google Drive folder."""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
            pageSize=1000, # Adjust if you have more than 1000 photos per folder
            fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception as e:
        print(f"[ERROR] Could not list files for folder {folder_id}: {e}")
        return []

def get_all_photo_files():
    """
    NEW FUNCTION
    Gets a flat list of all photo files from all session folders in Google Drive.
    This is what the worker will iterate over.
    """
    print("--- Fetching list of all photos from all Google Drive folders... ---")
    service = get_drive_service()
    all_folder_ids = get_all_gdrive_folder_ids()
    all_photos = []
    
    for folder_id in all_folder_ids:
        photos_in_folder = list_photo_links_from_folder(service, folder_id)
        print(f"  > Found {len(photos_in_folder)} photos in folder {folder_id}.")
        all_photos.extend(photos_in_folder)
        
    print(f"--- Total photos to process: {len(all_photos)} ---")
    return all_photos

def download_drive_photo(service, file_id):
    """Downloads photo data from Google Drive as a numpy array."""
    request = service.files().get_media(fileId=file_id)
    from io import BytesIO
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    file_bytes = np.asarray(bytearray(fh.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def compare_faces(face1_img, face2_img, threshold=0.593):
    """
    MODIFIED FUNCTION (Formerly is_face_match)
    Compares two cropped face images (numpy arrays).
    `enforce_detection` is False because faces are already detected.
    """
    try:
        result = DeepFace.verify(
            img1_path=face1_img, 
            img2_path=face2_img,
            model_name="SFace",
            enforce_detection=False  # Important: set to False
        )
        distance = result['distance']
        is_match = distance <= threshold
        print(f"    - Comparison distance: {distance:.4f}. Match: {is_match}")
        return is_match
    except Exception as e:
        print(f"    - [ERROR] DeepFace comparison failed: {e}")
        return False