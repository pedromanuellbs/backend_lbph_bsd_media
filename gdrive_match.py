# gdrive_match.py (MODIFIKASI UNTUK LBPH COMPARISON)

import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import cv2
import numpy as np
# from deepface import DeepFace # HAPUS BARIS INI

# Asumsi face_preprocessing sudah diimpor dan punya detect_and_crop
# dari proyek Anda, pastikan ini tersedia di lingkungan worker
from face_preprocessing import detect_and_crop

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

# --- Tambahkan ini di gdrive_match.py ---
# Ini harus sesuai dengan lokasi model dan label Anda
MODEL_PATH = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

def load_lbph_model_and_labels():
    """
    Load model LBPH dan label_map (lbl→user_id) dari filesystem.
    Ini adalah duplikasi dari app.py, pastikan model ter-training.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_MAP_PATH):
        print(f"ERROR: Model LBPH atau label map tidak ditemukan di {MODEL_PATH} atau {LABELS_MAP_PATH}")
        return None, {}

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)

    label_map = {}
    with open(LABELS_MAP_PATH, "r") as f:
        for line in f:
            lbl, uid = line.strip().split(":")
            label_map[int(lbl)] = uid
    return model, label_map

# Global variable untuk menyimpan model yang sudah di-load (opsional, untuk efisiensi)
_lbph_model = None
_lbph_label_map = None

def get_lbph_model():
    """Lazily load LBPH model and label map."""
    global _lbph_model, _lbph_label_map
    if _lbph_model is None or _lbph_label_map is None:
        _lbph_model, _lbph_label_map = load_lbph_model_and_labels()
    return _lbph_model, _lbph_label_map

def compare_faces(client_face_np_array, gdrive_face_np_array, confidence_threshold=70.0):
    """
    Fungsi yang dimodifikasi: Menggunakan LBPH untuk perbandingan wajah.
    Input adalah NumPy array gambar wajah yang sudah di-crop dan grayscale.
    """
    try:
        model, label_map = get_lbph_model()
        if model is None:
            print("    - [ERROR] Model LBPH tidak dimuat. Pastikan model sudah dilatih.")
            return False

        # Pastikan input adalah gambar grayscale dan bukan None
        if client_face_np_array is None or gdrive_face_np_array is None:
            print("    - [ERROR] Salah satu input wajah (NumPy array) adalah None.")
            return False
        
        # Pastikan gambar dalam format yang benar (CV_8UC1 - grayscale)
        # Jika detect_and_crop mengembalikan BGR, perlu dikonversi
        if len(client_face_np_array.shape) == 3 and client_face_np_array.shape[2] == 3:
            client_face_np_array = cv2.cvtColor(client_face_np_array, cv2.COLOR_BGR2GRAY)
        if len(gdrive_face_np_array.shape) == 3 and gdrive_face_np_array.shape[2] == 3:
            gdrive_face_np_array = cv2.cvtColor(gdrive_face_np_array, cv2.COLOR_BGR2GRAY)

        # Prediksi wajah klien untuk mendapatkan labelnya
        # Asumsi client_face_np_array sudah dari hasil detect_and_crop dan siap untuk predict
        predicted_client_label, client_confidence = model.predict(client_face_np_array)
        client_user_id_from_model = label_map.get(predicted_client_label)

        # Prediksi wajah dari Google Drive
        # Asumsi gdrive_face_np_array sudah dari hasil detect_and_crop dan siap untuk predict
        predicted_gdrive_label, gdrive_confidence = model.predict(gdrive_face_np_array)
        gdrive_user_id_from_model = label_map.get(predicted_gdrive_label)

        print(f"    - Klien (predict): ID={client_user_id_from_model}, Confidence={client_confidence:.2f}")
        print(f"    - GDrive (predict): ID={gdrive_user_id_from_model}, Confidence={gdrive_confidence:.2f}")

        # Kondisi cocok:
        # 1. User ID yang diprediksi dari wajah GDrive sama dengan user ID klien.
        # 2. Tingkat kepercayaan (confidence) untuk wajah GDrive berada di bawah threshold (semakin kecil semakin percaya diri).
        #    Kita juga bisa menambahkan cek confidence untuk wajah klien sendiri agar yakin model mengenalnya dengan baik.
        is_match = (gdrive_user_id_from_model == client_user_id_from_model) and \
                   (gdrive_confidence < confidence_threshold) 

        print(f"    - Hasil perbandingan LBPH: {is_match} (Confidence GDrive: {gdrive_confidence:.2f} < Threshold: {confidence_threshold:.2f})")
        return is_match

    except Exception as e:
        print(f"    - [ERROR] Perbandingan LBPH gagal: {e}")
        # import traceback; traceback.print_exc() # Aktifkan untuk debugging lebih lanjut
        return False