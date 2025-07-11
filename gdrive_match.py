# gdrive_match.py (Final LBPH)

import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import firebase_admin
from firebase_admin import firestore # Diperlukan untuk get_all_gdrive_folder_ids
import cv2
import numpy as np

# Pastikan face_preprocessing diimpor jika fungsi detect_and_crop ada di sana
# Jika detect_and_crop ada di face_preprocessing.py, pastikan baris ini diaktifkan:
# from face_preprocessing import detect_and_crop

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- Konstanta untuk Model LBPH ---
# Ini harus sesuai dengan lokasi model dan label Anda di lingkungan deployment
MODEL_PATH = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

# Global variable untuk menyimpan model yang sudah di-load (untuk efisiensi)
_lbph_model = None
_lbph_label_map = None

def get_drive_service():
    """Membuat objek layanan untuk Google Drive API."""
    cred_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not cred_json:
        raise ValueError("Variabel lingkungan 'GOOGLE_APPLICATION_CREDENTIALS_JSON' tidak diatur.")
    
    creds_info = json.loads(cred_json)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def get_all_gdrive_folder_ids():
    """Mengambil semua ID folder Google Drive dari koleksi 'photo_sessions' di Firestore."""
    print("\n--- Memulai get_all_gdrive_folder_ids ---")
    db = firestore.client()
    folder_ids = []
    
    try:
        sessions_stream = db.collection('photo_sessions').stream()
        sessions = list(sessions_stream)
        print(f"  > Ditemukan {len(sessions)} dokumen di koleksi 'photo_sessions'.")

        for doc in sessions:
            data = doc.to_dict()
            drive_link = data.get('driveLink', '')
            if 'folders/' in drive_link:
                folder_id = drive_link.split('folders/')[1].split('?')[0]
                folder_ids.append(folder_id)
    except Exception as e:
        print(f"  [ERROR] Terjadi exception saat mengambil data dari Firestore: {e}")

    print(f"  > Fungsi selesai. Daftar folder_ids yang akan dikembalikan: {folder_ids}")
    print("--- Selesai get_all_gdrive_folder_ids ---\n")
    return folder_ids

def list_photo_links_from_folder(service, folder_id):
    """Mendapatkan daftar metadata foto dari folder Google Drive tertentu."""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
            pageSize=1000,
            fields="files(id, name)").execute()
        return results.get('files', [])
    except Exception as e:
        print(f"[ERROR] Gagal mencantumkan file untuk folder {folder_id}: {e}")
        return []

def get_all_photo_files():
    """
    Mendapatkan daftar datar semua file foto dari semua folder sesi di Google Drive.
    Ini adalah apa yang akan diulang oleh worker.
    """
    print("--- Mengambil daftar semua foto dari semua folder Google Drive... ---")
    service = get_drive_service()
    all_folder_ids = get_all_gdrive_folder_ids()
    all_photos = []
    
    for folder_id in all_folder_ids:
        photos_in_folder = list_photo_links_from_folder(service, folder_id)
        print(f"  > Ditemukan {len(photos_in_folder)} foto di folder {folder_id}.")
        all_photos.extend(photos_in_folder)
        
    print(f"--- Total foto yang akan diproses: {len(all_photos)} ---")
    return all_photos

def download_drive_photo(service, file_id):
    """Mengunduh data foto dari Google Drive sebagai array numpy."""
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

    # Jika gambar gagal di-decode (misal: file corrupt total)
    if img is None:
        return None
    
    # Konversi paksa gambar ke format standar 8-bit untuk menghindari error VDepth
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    
    return img

def load_lbph_model_and_labels():
    """
    Memuat model LBPH dan label_map (lbl→user_id) dari sistem file.
    Pastikan model sudah dilatih dan file-nya ada di lokasi yang benar.
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

def get_lbph_model():
    """Memuat model LBPH dan peta label secara malas (lazy loading)."""
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
        predicted_client_label, client_confidence = model.predict(client_face_np_array)
        client_user_id_from_model = label_map.get(predicted_client_label)

        # Prediksi wajah dari Google Drive
        predicted_gdrive_label, gdrive_confidence = model.predict(gdrive_face_np_array)
        gdrive_user_id_from_model = label_map.get(predicted_gdrive_label)

        print(f"    - Klien (predict): ID={client_user_id_from_model}, Confidence={client_confidence:.2f}")
        print(f"    - GDrive (predict): ID={gdrive_user_id_from_model}, Confidence={gdrive_confidence:.2f}")

        # Kondisi cocok:
        # 1. User ID yang diprediksi dari wajah GDrive sama dengan user ID klien.
        # 2. Tingkat kepercayaan (confidence) untuk wajah GDrive berada di bawah threshold (semakin kecil semakin percaya diri).
        is_match = (gdrive_user_id_from_model == client_user_id_from_model) and \
                   (gdrive_confidence < confidence_threshold) 

        print(f"    - Hasil perbandingan LBPH: {is_match} (Confidence GDrive: {gdrive_confidence:.2f} < Threshold: {confidence_threshold:.2f})")
        return is_match

    except Exception as e:
        print(f"    - [ERROR] Perbandingan LBPH gagal: {e}")
        # import traceback; traceback.print_exc() # Aktifkan untuk debugging lebih lanjut
        return False

