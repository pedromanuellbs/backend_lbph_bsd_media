import os
import json
# import logging
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import firebase_admin
from firebase_admin import firestore
import cv2
import numpy as np
from facenet_pytorch import MTCNN

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Inisialisasi MTCNN untuk deteksi wajah
# mtcnn = MTCNN(keep_all=False, device='cpu')

# --- PERUBAHAN DI SINI: HAPUS BLOK INISIALISASI FIREBASE ---
# Blok 'if not firebase_admin._apps:' telah dihapus dari sini
# karena inisialisasi akan ditangani sepenuhnya oleh app.py
# -------------------------------------------------------------
#testing123

# Di file: gdrive_match.py

def get_all_gdrive_folder_ids():
    print("\n--- Memulai get_all_gdrive_folder_ids ---") # Laporan Awal
    db = firestore.client()
    folder_ids = []
    
    try:
        sessions_stream = db.collection('photo_sessions').stream()
        sessions = list(sessions_stream) # Ambil semua dokumen sekaligus
        print(f"  > Ditemukan {len(sessions)} dokumen di koleksi 'photo_sessions'.")

        for doc in sessions:
            print(f"  -> Memproses Dokumen ID: {doc.id}")
            data = doc.to_dict()
            drive_link = data.get('driveLink', '')
            print(f"     - Link Drive ditemukan: {drive_link}")

            # Cek apakah 'folders/' ada di dalam link
            if 'folders/' in drive_link:
                print("     - Kondisi 'folders/ in drive_link' terpenuhi (True).")
                folder_id = drive_link.split('folders/')[1].split('?')[0]
                folder_ids.append({
                    'sessionId': doc.id,       # Ini ID Dokumen Firestore (YANG BENAR)
                    'folderId': folder_id})
                print(f"     - ID Folder berhasil diekstrak: {folder_id}")
            else:
                print("     - Kondisi 'folders/ in drive_link' TIDAK terpenuhi (False). Melewati...")
    
    except Exception as e:
        print(f"  [ERROR] Terjadi exception saat mengambil data dari Firestore: {e}")

    print(f"  > Fungsi selesai. Daftar folder_ids yang akan dikembalikan: {folder_ids}")
    print("--- Selesai get_all_gdrive_folder_ids ---\n")
    return folder_ids

def get_drive_service():
    cred_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
    creds = service_account.Credentials.from_service_account_info(json.loads(cred_json), scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

def list_photos(folder_id):
    service = get_drive_service()
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'",
        pageSize=1000,
        fields="files(id, name, webViewLink)").execute() # Hanya butuh id, name, webViewLink
    return results.get('files', [])

def list_photo_links(folder_id):
    return list_photos(folder_id)

# Di file: gdrive_match.py

def download_drive_photo(file_id):
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

    # --- TAMBAHAN SOLUSI DI SINI ---
    # Jika gambar gagal di-decode (misal: file corrupt total)
    if img is None:
        return None
    
    # Konversi paksa gambar ke format standar 8-bit untuk menghindari error VDepth
    # Ini akan "membersihkan" gambar sebelum diproses lebih lanjut.
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    # ---------------------------------

    return img

# Di file: gdrive_match.py

def detect_and_crop_face(img):
    # --- FIX TERAKHIR: Paksa buat salinan gambar di memori ---
    # Bertujuan untuk memutus rantai cache/referensi objek yang mungkin terjadi.
    img_copy = img.copy()
    # ----------------------------------------------------------

    # Inisialisasi MTCNN di sini agar selalu fresh
    mtcnn = MTCNN(keep_all=False, device='cpu', post_process=False)
    
    # Gunakan img_copy untuk semua proses selanjutnya
    rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)
    if faces is None:
        return None
        
    if isinstance(faces, list) or len(faces.shape) == 4:
        face = faces[0]
    else:
        face = faces
        
    face_np = face.permute(1,2,0).byte().numpy()
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    
    return face_np

# Di file: gdrive_match.py

# ini sebelum (jam 5 sore 9 juli 2025)
def is_face_match(face_img, target_img, threshold=0.8): # Threshold baru, misal 0.8
    print("--- Memulai is_face_match (Logika Baru: HISTOGRAM) ---")
    
    face1 = detect_and_crop_face(face_img)
    face2 = detect_and_crop_face(target_img)
    
    if face1 is None or face2 is None:
        if face1 is None: print("  > Deteksi wajah klien (face1): Gagal")
        if face2 is None: print("  > Deteksi wajah target (face2): Gagal")
        print("--- Selesai is_face_match ---\n")
        return False

    print("  > Deteksi wajah klien (face1): Berhasil")
    print("  > Deteksi wajah target (face2): Berhasil")
    
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    
    # --- LOGIKA BARU: PERBANDINGAN HISTOGRAM ---
    # Hitung histogram untuk kedua gambar
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalisasi histogram agar skalanya sama
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Bandingkan histogram menggunakan metode korelasi
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # --- LOGIKA BARU SELESAI ---
    
    print(f"  > Skor Kemiripan (Korelasi Histogram): {score:.4f}")
    print(f"  > Ambang Batas (Threshold): {threshold}")
    
    # Logika baru: Dianggap cocok jika skor korelasi > threshold
    is_match = score > threshold
    
    print(f"  > Hasil Perbandingan: {'COCOK' if is_match else 'TIDAK COCOK'}")
    print("--- Selesai is_face_match ---\n")
    
    return is_match
    # # Lakukan prediksi untuk mendapatkan confidence score
    # label, conf = lbph_model.predict(gray2)
    
    # print(f"  > Skor Kemiripan (Confidence): {conf:.2f}") # Cetak skornya
    # print(f"  > Ambang Batas (Threshold): {threshold}") # Cetak threshold
    
    # is_match = conf < threshold
    # print(f"  > Hasil Perbandingan: {'COCOK' if is_match else 'TIDAK COCOK'}")
    # print("--- Selesai is_face_match ---\n")
    
    # return is_match

# Di file: gdrive_match.py

def find_matching_photos(user_face_path, session_id, folder_id, threshold=70):
    # Membaca gambar wajah user sekali saja di awal
    user_img = cv2.imread(user_face_path)
    if user_img is None:
        print(f"Error: Gagal membaca file wajah user di {user_face_path}")
        return []

    photos_in_folder = list_photo_links(folder_id)
    matched_in_folder = []

    print(f"Memeriksa {len(photos_in_folder)} foto di folder {folder_id}...")

    for photo in photos_in_folder:
        try:
            print(f"  -> Memproses foto: {photo['name']} ({photo['id']})")
            # 1. Download foto dari Google Drive
            target_img = download_drive_photo(photo['id'])
            if target_img is None:
                continue

            # 2. Lakukan perbandingan wajah
            # Fungsi is_face_match dipanggil di sini!
            if is_face_match(user_img, target_img, threshold):
                print(f"    [COCOK] Wajah ditemukan di foto {photo['name']}")
                # 3. Jika cocok, baru tambahkan ke daftar hasil

                file_id = photo['id']
                public_image_url = f'https://drive.google.com/uc?export=view&id={file_id}'


                matched_in_folder.append({
                    'name': photo['name'],
                    'webViewLink': photo['webViewLink'],
                    'webContentLink': public_image_url, # Gunakan link baru
                    'thumbnailLink': public_image_url,  # Gunakan link baru
                    'sessionId': session_id
                })
            else:
                print(f"    [TIDAK COCOK] Wajah tidak cocok di foto {photo['name']}")

        except Exception as e:
            print(f"    Error saat memproses foto {photo['name']}: {e}")
            continue
            
    return matched_in_folder

# def find_all_matching_photos(user_face_path, all_folder_ids, threshold=70):
#     all_matches = []
#     for folder_id in all_folder_ids:
#         matches = find_matching_photos(user_face_path, folder_id, threshold)
#         all_matches.extend(matches)
#     return all_matches

def find_all_matching_photos(user_face_path, all_sessions_data, threshold=70):
    all_matches = []
    for session in all_sessions_data:
        session_id = session['sessionId']
        folder_id = session['folderId']
        # Kirim kedua ID ke fungsi pencocokan
        matches = find_matching_photos(user_face_path, session_id, folder_id, threshold)
        all_matches.extend(matches)
    return all_matches
