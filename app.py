import os
import traceback
import json
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory

# Pastikan import ini sesuai dengan file Anda
from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

import firebase_admin
from firebase_admin import credentials, storage, firestore

# --- Inisialisasi Aplikasi ---
app = Flask(__name__)

# --- Konfigurasi dan Inisialisasi Firebase ---
try:
    print("[INFO] Inisialisasi Firebase Admin SDK...")
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'storageBucket': "db-ta-bsd-media.firebasestorage.app"
        })
    db = firestore.client()
    bucket = storage.bucket()
    print("[INFO] Firebase Admin SDK berhasil diinisialisasi.")
except Exception as e:
    print(f"[FATAL] Gagal inisialisasi Firebase: {e}")
    traceback.print_exc()

# --- Konstanta ---
DRIVE_API_KEY = "AIzaSyC_vPd6yPwYQ60Pn-tuR3Nly_7mgXZcxGk"

# --- Helper Functions ---
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP):
        return None, {}
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    label_map = {}
    with open(LABEL_MAP, "r") as f:
        for line in f:
            lbl, uid = line.strip().split(":")
            label_map[int(lbl)] = uid
    return model, label_map

def fetch_images_from_drive_folder(folder_url):
    urls = []
    folder_id = None
    if '/d/' in folder_url: folder_id = folder_url.split('/d/')[1].split('/')[0]
    elif '/folders/' in folder_url: folder_id = folder_url.split('/folders/')[1].split('/')[0]
    if not folder_id: return []
    
    api_url = "https://www.googleapis.com/drive/v3/files"
    params = {'q': f"'{folder_id}' in parents and mimeType contains 'image/'", 'fields': 'files(id)', 'key': DRIVE_API_KEY, 'pageSize': 100}
    try:
        response = requests.get(api_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        for file_info in data.get('files', []):
            urls.append(f"https://drive.google.com/uc?export=download&id={file_info['id']}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Gagal mengakses Google Drive API: {e}")
    return urls

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend (Haar Cascade Version) - Siap!"

@app.route('/register_face', methods=['POST'])
def register_face():
    # --- UBAHAN DI SINI ---
    # Logika training yang berat dihapus dari sini
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    if not user_id or not image: return jsonify({'success': False, 'error': 'user_id atau image tidak ada'}), 400
    
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')])
    raw_path = os.path.join(user_dir, f"{count+1}.jpg")
    image.save(raw_path)

    cropped = detect_and_crop(raw_path)
    if cropped is None: return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400
    
    cv2.imwrite(raw_path, cropped)
    
    # train_and_evaluate() # <-- BARIS INI DIHAPUS/DIKOMENTARI
    
    print(f"Wajah untuk user '{user_id}' berhasil disimpan. Jalankan /train_model untuk melatih ulang.")
    return jsonify({'success': True, 'message': 'Wajah berhasil disimpan. Model perlu di-train ulang secara manual.'})

# --- ENDPOINT BARU UNTUK TRAINING ---
@app.route('/train_model', methods=['GET'])
def train_model_endpoint():
    print("[INFO] Memulai proses training model...")
    try:
        metrics = train_and_evaluate()
        print("[SUCCESS] Model berhasil di-train ulang.")
        return jsonify({'success': True, 'message': 'Model berhasil di-train ulang.', 'metrics': metrics})
    except Exception as e:
        print(f"[FATAL] Gagal saat training model: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/find_my_photos', methods=['POST'])
def find_my_photos():
    # ... (isi fungsi ini tetap sama)
    print("\n[INFO] Memulai /find_my_photos...")
    image = request.files.get('image')
    if not image: return jsonify({'success': False, 'error': 'image tidak ada'}), 400
    tmp_path = 'tmp_find.jpg'; image.save(tmp_path)
    client_face_gray = detect_and_crop(tmp_path); os.remove(tmp_path)
    if client_face_gray is None: return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400
    model, label_map = load_model_and_labels()
    if model is None: return jsonify({'success': False, 'error': 'Model belum ada'}), 400
    label, confidence = model.predict(client_face_gray)
    client_user_id = label_map.get(label, 'unknown')
    print(f"[INFO] Wajah klien terverifikasi sebagai: '{client_user_id}' (Confidence: {confidence})")
    if client_user_id == 'unknown' or confidence > 80: return jsonify({'success': True, 'user_id': 'unknown', 'photo_urls': []})
    matching_urls = []
    sessions_ref = db.collection('photo_sessions').stream()
    for session in sessions_ref:
        session_data = session.to_dict()
        drive_link = session_data.get('driveLink')
        if not drive_link: continue
        print(f"[INFO] Memeriksa sesi: '{session_data.get('title')}'")
        photo_urls_in_drive = fetch_images_from_drive_folder(drive_link)
        for photo_url in photo_urls_in_drive:
            try:
                response = requests.get(photo_url, stream=True, timeout=15)
                if response.status_code == 200:
                    with open("temp_drive_photo.jpg", 'wb') as f: f.write(response.content)
                    drive_photo_gray = detect_and_crop("temp_drive_photo.jpg")
                    if drive_photo_gray is not None:
                        predicted_label, pred_conf = model.predict(drive_photo_gray)
                        predicted_user_id = label_map.get(predicted_label)
                        if predicted_user_id == client_user_id and pred_conf < 80: 
                            print(f"[SUCCESS] COCOK! Foto {photo_url.split('&id=')[1]} adalah milik '{client_user_id}' (Conf: {pred_conf})")
                            matching_urls.append(photo_url)
            except Exception as e:
                print(f"[ERROR] Gagal memproses foto dari drive {photo_url}: {e}")
        if os.path.exists("temp_drive_photo.jpg"): os.remove("temp_drive_photo.jpg")
    return jsonify({'success': True, 'user_id': client_user_id, 'photo_urls': matching_urls})

# --- Menjalankan Server ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
