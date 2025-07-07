import os
import json
from io import BytesIO

import cv2
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from facenet_pytorch import MTCNN

import firebase_admin
from firebase_admin import firestore

# Konfigurasi
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
THRESHOLD_LBPH = 70  # ambang batas LBPH: semakin kecil semakin ketat

# Inisialisasi MTCNN untuk deteksi wajah (CPU)
mtcnn = MTCNN(keep_all=False, device='cpu')

# Utility: ambil instance Firestore
def get_firestore_client():
    return firestore.client()

# Utility: inisialisasi Drive API
def get_drive_service():
    cred_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not cred_json:
        raise RuntimeError('Env GOOGLE_APPLICATION_CREDENTIALS_JSON tidak ditemukan')
    creds = service_account.Credentials.from_service_account_info(
        json.loads(cred_json), scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

# Ambil semua folder Google Drive fotografer dari Firestore
def get_all_gdrive_folder_ids():
    db = get_firestore_client()
    sessions = db.collection('photo_sessions').stream()
    folder_ids = []
    for doc in sessions:
        data = doc.to_dict()
        link = data.get('driveLink', '')
        if 'folders/' in link:
            fid = link.split('folders/')[1].split('?')[0]
            folder_ids.append(fid)
    return folder_ids

# List metadata foto di folder
def list_photos(folder_id):
    service = get_drive_service()
    q = f"'{folder_id}' in parents and trashed = false and mimeType contains 'image/'"
    resp = service.files().list(
        q=q,
        pageSize=1000,
        fields='files(id, name, webViewLink, thumbnailLink)'
    ).execute()
    return resp.get('files', [])

# Download gambar dari Drive sebagai OpenCV image
def download_drive_photo(file_id):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    arr = np.frombuffer(fh.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# Deteksi dan crop wajah dari gambar
def detect_and_crop_face(img):
    # MTCNN output: Tensor [C,H,W]
    faces = mtcnn(img)
    if faces is None:
        return None
    face = faces[0] if isinstance(faces, list) else faces
    face_np = face.permute(1, 2, 0).int().numpy()
    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return face_bgr

# Cek kecocokan dua wajah dengan LBPH
def is_face_match(user_img, target_img, lbph_model, threshold=THRESHOLD_LBPH):
    f1 = detect_and_crop_face(user_img)
    f2 = detect_and_crop_face(target_img)
    if f1 is None or f2 is None:
        return False
    gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    label, conf = lbph_model.predict(gray)
    # conf: semakin kecil semakin mirip
    return conf < threshold

# Temukan foto cocok dalam satu folder
def find_matching_photos(user_face_path, folder_id, lbph_model, threshold=THRESHOLD_LBPH):
    # Load gambar user
    user_img = cv2.imread(user_face_path)
    if user_img is None:
        raise FileNotFoundError(f'File user tidak ditemukan: {user_face_path}')

    matches = []
    for meta in list_photos(folder_id):
        try:
            target_img = download_drive_photo(meta['id'])
            if is_face_match(user_img, target_img, lbph_model, threshold):
                matches.append({
                    'name': meta['name'],
                    'webViewLink': meta['webViewLink'],
                    'thumbnailLink': meta['thumbnailLink'],
                })
        except Exception as e:
            # bisa log error jika perlu
            continue
    return matches

# Temukan di semua folder fotografer
def find_all_matching_photos(user_face_path, lbph_model, threshold=THRESHOLD_LBPH):
    folder_ids = get_all_gdrive_folder_ids()
    all_matches = []
    for fid in folder_ids:
        ms = find_matching_photos(user_face_path, fid, lbph_model, threshold)
        all_matches.extend(ms)
    return all_matches

# Contoh pemakaian (Dynamic Input)

# 1. Melalui Command-Line
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Matching foto hasil scan pengguna'
    )
    parser.add_argument(
        'user_face_path',
        help='Path ke foto hasil scan klien yang dikirim dari aplikasi Flutter'
    )
    args = parser.parse_args()

    user_path = args.user_face_path

    # Inisialisasi Firestore (sudah dilakukan di app.py)
    db = firestore.client()

    # Buat model LBPH di OpenCV
    lbph = cv2.face.LBPHFaceRecognizer_create()
    # Jika ada model terlatih:
    # lbph.read('lbph_model.xml')

    matches = find_all_matching_photos(user_path, lbph)
    print(f'Jumlah foto cocok: {len(matches)}')
    for m in matches:
        print(m['webViewLink'])

# 2. Contoh Endpoint HTTP (Flask)
# Aplikasi Flutter dapat mengupload file wajah ke endpoint ini.
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/match', methods=['POST'])
def match_endpoint():
    # File di-form field 'face'
    file = request.files.get('face')
    if not file:
        return jsonify({'error': 'Tidak ada file wajah'}), 400

    # Baca langsung sebagai array numpy via OpenCV
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    user_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Inisialisasi LBPH (pastikan lbph dan model sudah diinisialisasi di luar)
    lbph = cv2.face.LBPHFaceRecognizer_create()
    # lbph.read('lbph_model.xml')  # jika menggunakan model terlatih

    # Pencocokan
    # Kita bisa membuat fungsi find_all_matching_photos_bytes untuk menerima image array
    matches = find_all_matching_photos_bytes(user_img, lbph)

    urls = [m['webViewLink'] for m in matches]
    return jsonify({'matches': urls})