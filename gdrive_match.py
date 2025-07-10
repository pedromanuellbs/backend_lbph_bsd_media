import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import cv2
import numpy as np
from deepface import DeepFace # <-- PUSTAKA BARU

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_all_gdrive_folder_ids():
    """
    Mengambil semua ID folder dari koleksi 'photo_sessions' di Firestore.
    Fungsi ini tidak berubah.
    """
    print("\n--- Memulai get_all_gdrive_folder_ids ---")
    # Impor firestore di dalam fungsi untuk menghindari masalah inisialisasi parsial
    from firebase_admin import firestore
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
                folder_ids.append({
                    'sessionId': doc.id,
                    'folderId': folder_id
                })
    except Exception as e:
        print(f"  [ERROR] Terjadi exception saat mengambil data dari Firestore: {e}")

    print(f"  > Fungsi selesai. Daftar folder_ids: {folder_ids}")
    print("--- Selesai get_all_gdrive_folder_ids ---\n")
    return folder_ids

def get_drive_service():
    """
    Membuat service untuk otentikasi dengan Google Drive API.
    Fungsi ini tidak berubah.
    """
    cred_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
    creds = service_account.Credentials.from_service_account_info(json.loads(cred_json), scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_photo_links(folder_id):
    """
    Mendapatkan daftar metadata foto dari folder Google Drive.
    Fungsi ini tidak berubah.
    """
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
        pageSize=1000,
        fields="files(id, name, webViewLink)").execute()
    return results.get('files', [])

def download_drive_photo(file_id):
    """
    Mengunduh data gambar dari Google Drive berdasarkan file_id.
    Fungsi ini tidak berubah.
    """
    service = get_drive_service()
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

    if img is None:
        return None
    
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img

# --- PERUBAHAN UTAMA DIMULAI DI SINI ---

# Di gdrive_match.py
def is_face_match(user_face_path, target_face_path, threshold=0.4): # Tambahkan parameter
    """
    Memverifikasi apakah dua gambar wajah adalah orang yang sama menggunakan DeepFace.
    """
    print(f"--- Memulai is_face_match (DeepFace) ---")
    try:
        result = DeepFace.verify(
            img1_path=user_face_path, 
            img2_path=target_face_path,
            model_name="VGG-Face",
            enforce_detection=True
        )
        
        distance = result['distance']
        # LOGIKA BARU: Jarak harus LEBIH KECIL dari atau sama dengan threshold
        is_match = distance <= threshold
        
        print(f"  > Jarak: {distance:.4f}, Ambang Batas: {threshold}")
        print(f"  > Hasil Verifikasi: {'COCOK' if is_match else 'TIDAK COCOK'}")
        print("--- Selesai is_face_match ---\n")
        return is_match

    except Exception as e:
        print(f"  > Error DeepFace: {e}")
        print("--- Selesai is_face_match ---\n")
        return False

def find_matching_photos(user_face_path, session_id, folder_id, threshold=0.4): # Tambahkan parameter
    """
    Mencari foto yang cocok di dalam satu folder Google Drive.
    Fungsi ini disesuaikan untuk bekerja dengan is_face_match yang baru.
    """
    photos_in_folder = list_photo_links(folder_id)
    matched_in_folder = []

    print(f"Memeriksa {len(photos_in_folder)} foto di folder {folder_id}...")
    
    # Definisikan path untuk menyimpan file target sementara
    tmp_target_path = "tmp_target_image.jpg"

    for photo in photos_in_folder:
        try:
            print(f"  -> Memproses foto: {photo['name']} ({photo['id']})")
            
            # 1. Download foto dari Google Drive
            target_img_data = download_drive_photo(photo['id'])
            if target_img_data is None:
                print(f"     - Gagal mengunduh atau decode foto {photo['name']}. Melewati.")
                continue

            # 2. Simpan gambar yang diunduh ke file sementara
            cv2.imwrite(tmp_target_path, target_img_data)

            # 3. Lakukan perbandingan wajah menggunakan path file
            if is_face_match(user_face_path, tmp_target_path):
                file_id = photo['id']
                public_image_url = f'https://drive.google.com/uc?export=view&id={file_id}'

                matched_in_folder.append({
                    'name': photo['name'],
                    'webViewLink': photo['webViewLink'],
                    'webContentLink': public_image_url,
                    'thumbnailLink': public_image_url,
                    'sessionId': session_id
                })
        
        except Exception as e:
            print(f"    Error saat memproses foto {photo['name']}: {e}")
            continue
    
    # Hapus file sementara setelah selesai loop
    if os.path.exists(tmp_target_path):
        os.remove(tmp_target_path)
            
    return matched_in_folder

def find_all_matching_photos(user_face_path, all_sessions_data, threshold=0.4): # Tambahkan parameter
    """
    Menjalankan pencarian di semua folder yang didapat dari Firestore.
    Fungsi ini tidak berubah, hanya memanggil find_matching_photos yang sudah disesuaikan.
    """
    all_matches = []
    for session in all_sessions_data:
        session_id = session['sessionId']
        folder_id = session['folderId']
        # Memanggil fungsi yang sudah di-update
        matches = find_matching_photos(user_face_path, session_id, folder_id)
        all_matches.extend(matches)
    return all_matches

# Catatan: Fungsi detect_and_crop_face(img) yang lama telah dihapus 
# karena sudah tidak diperlukan lagi. Logika deteksi ditangani oleh DeepFace.