import os
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import firebase_admin
from firebase_admin import firestore
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch # Diperlukan untuk inisialisasi MTCNN device

# Pastikan ini konsisten dengan config.py Anda
# Jika Anda mengimpor dari config, hapus baris ini dan gunakan:
# from config import MODEL_PATH, LABEL_MAP
MODEL_PATH = "lbph_model.xml" # Atau .yml jika itu format Anda
LABEL_MAP = "labels_map.txt"

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- Inisialisasi MTCNN yang konsisten ---
# Ini harusnya sama dengan yang di face_preprocessing.py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=96, margin=0, keep_all=False, post_process=True, device=device)
# ------------------------------------------

# --- GLOBAL: Muat model LBPH dan label map sekali ---
# Model global ini masih diperlukan untuk endpoint /register_face dan /verify_face di app.py
# yang menggunakan model LBPH utama untuk klasifikasi.
_lbph_model = None
_label_to_user_map = None # Peta dari label int ke user_id string

def _load_global_lbph_model():
    global _lbph_model, _label_to_user_map
    if _lbph_model is None:
        print("INFO: Memuat model LBPH global untuk gdrive_match.py...")
        _lbph_model = cv2.face.LBPHFaceRecognizer_create()
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP):
            print(f"ERROR: Model ({MODEL_PATH}) atau label map ({LABEL_MAP}) tidak ditemukan untuk gdrive_match.py.")
            _lbph_model = None # Pastikan model tetap None jika file tidak ada
            _label_to_user_map = {}
            return None, {} # Mengembalikan tuple untuk konsistensi

        try:
            _lbph_model.read(MODEL_PATH)
            print("INFO: Model LBPH global berhasil dimuat.")
        except cv2.error as e:
            print(f"CRITICAL ERROR: Gagal membaca model LBPH di gdrive_match.py: {e}")
            _lbph_model = None
            _label_to_user_map = {}
            return None, {}

        _label_to_user_map = {}
        try:
            with open(LABEL_MAP, "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        lbl, uid = parts
                        _label_to_user_map[int(lbl)] = uid
            print("INFO: Label map global berhasil dimuat.")
        except Exception as e:
            print(f"CRITICAL ERROR: Gagal memuat label map di gdrive_match.py: {e}")
            _lbph_model = None
            _label_to_user_map = {}
            return None, {}
    return _lbph_model, _label_to_user_map


def get_all_gdrive_folder_ids():
    print("\n--- Memulai get_all_gdrive_folder_ids ---")
    db = firestore.client()
    folder_ids = []
    
    try:
        sessions_stream = db.collection('photo_sessions').stream()
        sessions = list(sessions_stream)
        print(f"  > Ditemukan {len(sessions)} dokumen di koleksi 'photo_sessions'.")

        for doc in sessions:
            print(f"  -> Memproses Dokumen ID: {doc.id}")
            data = doc.to_dict()
            drive_link = data.get('driveLink', '')
            print(f"     - Link Drive ditemukan: {drive_link}")

            if 'folders/' in drive_link:
                print("     - Kondisi 'folders/ in drive_link' terpenuhi (True).")
                folder_id = drive_link.split('folders/')[1].split('?')[0]
                folder_ids.append(folder_id)
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
        fields="files(id, name, webViewLink, thumbnailLink, webContentLink)").execute()
    return results.get('files', [])

def list_photo_links(folder_id):
    return list_photos(folder_id)

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

    if img is None:
        print(f"WARNING: Gagal decode gambar dari Drive untuk file ID: {file_id}")
        return None
    
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
        print(f"DEBUG: Gambar dari Drive dikonversi ke format 8-bit untuk file ID: {file_id}")

    return img

def detect_and_crop_face(img):
    # Ini harusnya menggunakan MTCNN yang diinisialisasi secara global di file ini
    # dan konsisten dengan face_preprocessing.py
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)
    if faces is None:
        return None
    
    # MTCNN dapat mengembalikan tensor dengan dimensi batch (1, C, H, W) atau (C, H, W)
    if isinstance(faces, list) or faces.dim() == 4:
        face = faces[0] # Ambil wajah pertama jika ada banyak atau dalam list
    else:
        face = faces
    
    # Konversi tensor PyTorch ke NumPy array (H, W, C)
    # Pastikan tensor dipindahkan ke CPU sebelum konversi ke NumPy
    face_np = face.permute(1,2,0).to('cpu').numpy()
    # Konversi dari RGB (output MTCNN) ke BGR (untuk OpenCV)
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_np

def is_face_match(user_face_img, target_img, threshold=108): # <--- THRESHOLD DEFAULT DIUBAH KE 120
    print("--- Memulai is_face_match (Logika Perbandingan 1:1) ---")
    
    # Deteksi dan crop wajah dari FOTO KLIEN yang di-upload
    face1_cropped = detect_and_crop_face(user_face_img) # Wajah Klien (sudah di-crop dan distandardisasi)
    face2_cropped = detect_and_crop_face(target_img) # Wajah Target dari Drive (sudah di-crop dan distandardisasi)

    if face1_cropped is None:
        print("  > Deteksi wajah klien (face1): Gagal. Melewati perbandingan.")
        return False
    if face2_cropped is None:
        print("  > Deteksi wajah target (face2): Gagal. Melewati perbandingan.")
        return False
    
    print("  > Deteksi wajah klien (face1): Berhasil")
    print("  > Deteksi wajah target (face2): Berhasil")

    # Konversi ke Grayscale untuk LBPH
    gray1 = cv2.cvtColor(face1_cropped, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2_cropped, cv2.COLOR_BGR2GRAY)
    
    # --- LOGIKA PERBANDINGAN 1:1 DENGAN MODEL SEMENTARA ---
    # Buat model LBPH sementara
    temp_model = cv2.face.LBPHFaceRecognizer_create()

    # Latih model sementara HANYA dengan wajah klien
    # Beri label 0 untuk wajah klien
    temp_model.train([gray1], np.array([0]))

    # DEBUG: Prediksi wajah klien itu sendiri untuk melihat confidence
    label_self, conf_self = temp_model.predict(gray1)
    print(f"  > DEBUG: Klien (gray1) diprediksi terhadap dirinya sendiri: Label {label_self}, Conf {conf_self:.2f}")

    # Prediksi wajah target menggunakan model sementara ini
    # conf akan menjadi jarak antara wajah target dan wajah klien (label 0)
    label, conf = temp_model.predict(gray2)

    print(f"  > Skor Kemiripan (Confidence): {conf:.2f}")
    print(f"  > Ambang Batas (Threshold): {threshold}")

    # Logika perbandingan: conf < threshold (makin kecil conf, makin mirip)
    is_match = conf < threshold

    if is_match:
        print(f"  > Hasil Perbandingan: COCOK")
    else:
        print(f"  > Hasil Perbandingan: TIDAK COCOK")

    # --- DEBUGGING: Cek apakah confidence 0.00 untuk gambar berbeda ---
    # Ini adalah log untuk membantu debugging jika conf selalu 0.00.
    # Jika Anda melihat ini, itu menunjukkan masalah mendasar dengan LBPH.
    if conf == 0.00 and not np.array_equal(gray1, gray2):
        print("WARNING: Confidence is 0.00 for non-identical images. This indicates an issue with LBPH or input.")
    # --- END DEBUGGING ---

    print("--- Selesai is_face_match ---\n")
    
    return is_match

def find_matching_photos(user_face_path, folder_id, threshold=108): # <--- THRESHOLD DEFAULT DIUBAH KE 120
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
            target_img = download_drive_photo(photo['id'])
            if target_img is None:
                continue

            if is_face_match(user_img, target_img, threshold):
                print(f"    [COCOK] Wajah ditemukan di foto {photo['name']}")
                matched_in_folder.append({
                    'name': photo['name'],
                    'webViewLink': photo['webViewLink'],
                    'thumbnailLink': photo['thumbnailLink'],
                    'sessionId': folder_id,
                })
            else:
                print(f"    [TIDAK COCOK] Wajah tidak cocok di foto {photo['name']}")

        except Exception as e:
            print(f"    Error saat memproses foto {photo['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return matched_in_folder

def find_all_matching_photos(user_face_path, all_folder_ids, threshold=108): # <--- THRESHOLD DEFAULT DIUBAH KE 120
    all_matches = []
    for folder_id in all_folder_ids:
        matches = find_matching_photos(user_face_path, folder_id, threshold)
        all_matches.extend(matches)
    return all_matches
