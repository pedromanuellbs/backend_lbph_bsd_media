# 1. Setelah user klien scan wajah dan upload ke endpoint /find_my_photos
from googleapiclient.discovery import build
from google.oauth2 import service_account
import cv2
import numpy as np
import requests

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = '1vmuvTW...'  # ID Google Drive folder fotografer

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_photos(folder_id):
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
        pageSize=1000,
        fields="files(id, name, webViewLink, thumbnailLink)").execute()
    return results.get('files', [])

def download_drive_photo(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    from io import BytesIO
    from googleapiclient.http import MediaIoBaseDownload
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    # Convert to numpy array
    file_bytes = np.asarray(bytearray(fh.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def is_face_match(face_img, target_img, model):
    # Implement your face matching logic here (LBPH, cosine similarity, dlib, etc)
    # return True if match
    pass

def find_matching_photos(user_face_path, folder_id, model):
    user_img = cv2.imread(user_face_path)
    photos = list_photos(folder_id)
    matched = []
    for photo in photos:
        try:
            img = download_drive_photo(photo['id'])
            if is_face_match(user_img, img, model):
                matched.append({
                    'name': photo['name'],
                    'webViewLink': photo['webViewLink'],
                    'thumbnailLink': photo['thumbnailLink'],
                })
        except Exception as e:
            print(f"Error checking photo {photo['name']}: {e}")
    return matched


def find_all_matching_photos(user_face_path, all_folder_ids, lbph_model, threshold=70):
    all_matches = []
    for folder_id in all_folder_ids:
        matches = find_matching_photos(user_face_path, folder_id, lbph_model, threshold)
        all_matches.extend(matches)
    return all_matches