import os
import json
import cv2
import numpy as np
import firebase_admin
from firebase_admin import firestore
from facenet_pytorch import MTCNN
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO

# --- Fungsi Utilitas Google Drive & Firebase (TIDAK BERUBAH) ---

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_service():
    cred_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
    creds = service_account.Credentials.from_service_account_info(json.loads(cred_json), scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def download_drive_photo(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    file_bytes = np.asarray(bytearray(fh.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None: return None
    if img.dtype != np.uint8: img = cv2.convertScaleAbs(img)
    return img

def get_all_gdrive_folder_ids():
    db = firestore.client()
    folder_ids = []
    sessions = db.collection('photo_sessions').stream()
    for doc in sessions:
        drive_link = doc.to_dict().get('driveLink', '')
        if 'folders/' in drive_link:
            folder_id = drive_link.split('folders/')[1].split('?')[0]
            folder_ids.append(folder_id)
    return folder_ids

# --- FUNGSI UTAMA YANG DITULIS ULANG SECARA TOTAL ---

def find_matching_photos(user_face_path, folder_id, threshold=0.8):
    print(f"\nMemeriksa folder {folder_id} dengan logika baru...")
    
    # --- Bagian 1: Proses Wajah Klien (dilakukan sekali di awal) ---
    try:
        user_img = cv2.imread(user_face_path)
        if user_img is None:
            print(f"Error: Gagal membaca file wajah user di {user_face_path}")
            return []

        mtcnn_client = MTCNN(keep_all=False, device='cpu', post_process=False)
        face_client_cropped_bgr = mtcnn_client(cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB))
        
        if face_client_cropped_bgr is None:
            print("Gagal mendeteksi wajah pada FOTO KLIEN.")
            return []
            
        face_client_cropped_bgr = face_client_cropped_bgr.permute(1, 2, 0).byte().numpy()
        face_client_cropped_bgr = cv2.cvtColor(face_client_cropped_bgr, cv2.COLOR_RGB2BGR)

        gray_client = cv2.cvtColor(face_client_cropped_bgr, cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([gray_client], [0], None, [256], [0, 256])
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        print("Berhasil memproses wajah klien dan membuat histogram referensi.")
    except Exception as e:
        print(f"Error fatal saat memproses wajah klien: {e}")
        return []

    # --- Bagian 2: Loop dan Proses Setiap Foto Target ---
    photos_in_folder = list_photos(folder_id)
    matched_in_folder = []
    
    for photo in photos_in_folder:
        try:
            print(f"  -> Memproses foto target: {photo['name']}")
            target_img = download_drive_photo(photo['id'])
            if target_img is None: continue

            mtcnn_target = MTCNN(keep_all=False, device='cpu', post_process=False)
            face_target_cropped_bgr = mtcnn_target(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

            if face_target_cropped_bgr is None:
                print("    > Wajah tidak ditemukan di foto target.")
                continue
            
            face_target_cropped_bgr = face_target_cropped_bgr.permute(1, 2, 0).byte().numpy()
            face_target_cropped_bgr = cv2.cvtColor(face_target_cropped_bgr, cv2.COLOR_RGB2BGR)

            gray_target = cv2.cvtColor(face_target_cropped_bgr, cv2.COLOR_BGR2GRAY)
            hist2 = cv2.calcHist([gray_target], [0], None, [256], [0, 256])
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            print(f"    > Skor Korelasi: {score:.4f} | Threshold: {threshold}")
            
            if score > threshold:
                print(f"    > [COCOK] Menambahkan foto {photo['name']}")
                matched_in_folder.append({
                    'name': photo['name'],
                    'webViewLink': photo['webViewLink'],
                    'thumbnailLink': photo['thumbnailLink'],
                })
            else:
                print("    > [TIDAK COCOK]")

        except Exception as e:
            print(f"    > Terjadi error saat memproses foto {photo['name']}: {e}")
            continue
            
    return matched_in_folder

def find_all_matching_photos(user_face_path, all_folder_ids, threshold=0.8):
    all_matches = []
    for folder_id in all_folder_ids:
        matches = find_matching_photos(user_face_path, folder_id, threshold)
        all_matches.extend(matches)
    return all_matches