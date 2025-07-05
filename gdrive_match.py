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

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- PERUBAHAN DI SINI: Terapkan Lazy Loading untuk MTCNN ---

# 1. Buat variabel global untuk menyimpan model, awalnya kosong
mtcnn_model = None

def get_mtcnn():
    """
    Fungsi ini akan memuat model MTCNN jika belum ada, 
    dan mengembalikannya. Jika sudah ada, langsung kembalikan.
    """
    global mtcnn_model
    if mtcnn_model is None:
        print("Mulai memuat model MTCNN (pertama kali)...")
        # Inisialisasi MTCNN hanya terjadi satu kali di sini
        mtcnn_model = MTCNN(keep_all=False, device='cpu', post_process=False)
        print("Model MTCNN berhasil dimuat.")
    return mtcnn_model

# Hapus inisialisasi MTCNN dari sini
# mtcnn = MTCNN(keep_all=False, device='cpu') 

# --- AKHIR PERUBAHAN ---


if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = firebase_admin.credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred)

def get_all_gdrive_folder_ids():
    db = firestore.client()
    folder_ids = []
    sessions = db.collection('photo_sessions').stream()
    for doc in sessions:
        data = doc.to_dict()
        drive_link = data.get('driveLink', '')
        if 'folders/' in drive_link:
            folder_id = drive_link.split('folders/')[1].split('?')[0]
            folder_ids.append(folder_id)
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
        fields="files(id, name, webViewLink, thumbnailLink)").execute()
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
    return img

def detect_and_crop_face(img):
    # --- PERUBAHAN DI SINI: Panggil fungsi get_mtcnn ---
    mtcnn = get_mtcnn()
    # --- AKHIR PERUBAHAN ---
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)
    if faces is None:
        return None
    
    # MTCNN dengan post_process=False mengembalikan tensor, perlu diolah manual
    face = faces[0] 
    face_np = face.permute(1, 2, 0).int().numpy()
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_np

def is_face_match(face_img, target_img, lbph_model, threshold=70):
    face1 = detect_and_crop_face(face_img)
    face2 = detect_and_crop_face(target_img)
    print("Detect face user:", face1 is not None, "Detect face target:", face2 is not None)
    if face1 is None or face2 is None:
        return False
    gray1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    label, conf = lbph_model.predict(gray2)
    print("LBPH conf:", conf)
    return conf < threshold

def find_matching_photos(user_face_path, folder_id, lbph_model, threshold=70):
    photos = list_photo_links(folder_id)
    matched = []
    user_face_img = cv2.imread(user_face_path)
    if user_face_img is None:
        return []

    for photo in photos:
        print(f"Mencocokkan dengan {photo['name']}...")
        target_img = download_drive_photo(photo['id'])
        if target_img is not None and is_face_match(user_face_img, target_img, lbph_model, threshold):
            matched.append({
                'name': photo['name'],
                'webViewLink': photo['webViewLink'],
                'thumbnailLink': photo['thumbnailLink'],
            })
    return matched

def find_all_matching_photos(user_face_path, all_folder_ids, lbph_model, threshold=70):
    all_matches = []
    for folder_id in all_folder_ids:
        matches = find_matching_photos(user_face_path, folder_id, lbph_model, threshold)
        all_matches.extend(matches)
    return all_matches