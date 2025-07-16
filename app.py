import os
import sys
import json
import traceback
import tempfile
import numpy as np
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory
import time # Import time untuk timestamp

from face_preprocessing import detect_and_crop
# Import fungsi yang diperbarui dari face_data
from face_data import update_lbph_model_incrementally, train_and_evaluate_full_dataset, load_model_and_labels

from config import FACES_DIR, MODEL_PATH, LABEL_MAP # Pastikan ini mengarah ke file config Anda

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage, firestore

from gdrive_match import find_matching_photos, find_all_matching_photos, get_all_gdrive_folder_ids

# --- Inisialisasi Firebase ---
if not firebase_admin._apps:
    try:
        # Asumsi GOOGLE_APPLICATION_CREDENTIALS_JSON diset sebagai variabel lingkungan di Railway
        cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        cred = credentials.Certificate(cred_info)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
        })
        print("Firebase Admin SDK initialized successfully.")
    except KeyError:
        print("ERROR: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON not set.")
        print("Firebase Admin SDK will not be initialized. Check your Railway environment variables.")
    except Exception as e:
        print(f"ERROR initializing Firebase Admin SDK: {e}")
        traceback.print_exc()

# --- Variabel Global untuk Model ---
global_recognizer = None
global_labels_reverse = None

# --- Fungsi Helper ---
def upload_to_firebase(local_file, user_id, filename):
    """Upload file ke Firebase Storage dan return URL download-nya"""
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
    return blob.public_url

def download_file_from_url(url, destination):
    try:
        print(f"DEBUG: Mengunduh {url} ke {destination}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"DEBUG: Berhasil mengunduh {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Gagal mengunduh file dari URL {url}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Terjadi error saat menyimpan file {destination}: {e}")
        return False

# --- Fungsi Pemuatan Model ---
def load_models_globally():
    global global_recognizer, global_labels_reverse
    print("INFO: Memeriksa dan memuat model LBPH ke memori...")

    MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flbph_model.xml?alt=media&token=26656ed8-3cd1-4220-a07d-aad9aaeb91f5"
    LABEL_MAP_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flabels_map.txt?alt=media&token=2ab5957f-78b2-41b0-a1aa-b2f1b8675f54"

    # Untuk production dengan Docker + Volume, path ini akan merujuk ke volume
    # Untuk development lokal, pastikan direktori 'data' ada
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    MODEL_PATH_LOCAL = os.path.join(DATA_DIR, "lbph_model.xml")
    LABEL_MAP_LOCAL = os.path.join(DATA_DIR, "labels_map.txt")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH_LOCAL):
        print(f"INFO: Model tidak ditemukan di {MODEL_PATH_LOCAL}. Mengunduh...")
        if not download_file_from_url(MODEL_URL, MODEL_PATH_LOCAL):
            print("CRITICAL: Gagal mengunduh model.")
            return

    if not os.path.exists(LABEL_MAP_LOCAL):
        print(f"INFO: Label map tidak ditemukan di {LABEL_MAP_LOCAL}. Mengunduh...")
        if not download_file_from_url(LABEL_MAP_URL, LABEL_MAP_LOCAL):
            print("CRITICAL: Gagal mengunduh label map.")
            return
            
    try:
        global_recognizer = cv2.face.LBPHFaceRecognizer_create()
        global_recognizer.read(MODEL_PATH_LOCAL)

        labels = {}
        with open(LABEL_MAP_LOCAL, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ":" in line:
                    k, v = line.split(":", 1)
                    labels[v.strip()] = int(k.strip())
        
        global_labels_reverse = {v: k for k, v in labels.items()}
        print("SUCCESS: Model dan label berhasil dimuat.")
    except Exception as e:
        print(f"CRITICAL: Gagal memuat model/label dari file: {e}")
        traceback.print_exc()

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
os.makedirs(FACES_DIR, exist_ok=True)

@app.errorhandler(Exception)
def handle_exceptions(e):
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

# --- Endpoints / Routes ---

@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

@app.route('/face_login', methods=['POST'])
def face_login():
    print("REQUEST MASUK KE /face_login")
    
    claimed_username = request.form.get('user_id')
    image_file = request.files.get('image')

    if not claimed_username or not image_file:
        return jsonify({'error': 'user_id dan image wajib diisi'}), 400

    if global_recognizer is None or global_labels_reverse is None:
        return jsonify({'error': 'Server model is not ready, please try again later.'}), 503

    try:
        in_memory_file = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Format gambar tidak valid atau korup.'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            return jsonify({'status': 'not_found', 'message': 'Wajah tidak terdeteksi di foto.'}), 400

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        predicted_id, confidence = global_recognizer.predict(roi_gray)
        print(f"DEBUG: Claimed User: {claimed_username}, Predicted ID: {predicted_id}, Confidence: {confidence}")

        if confidence < 60 and predicted_id in global_labels_reverse:
            recognized_username = global_labels_reverse[predicted_id]
            print(f"DEBUG: Recognized User: {recognized_username}")

            if recognized_username == claimed_username:
                return jsonify({
                    'status': 'success',
                    'message': 'Login berhasil!',
                    'user': {'username': recognized_username, 'confidence': float(confidence)}
                }), 200
            else:
                return jsonify({'status': 'match_failed', 'message': 'Wajah Anda tidak cocok dengan username yang dimasukkan.'}), 403
        else:
            return jsonify({'status': 'not_found', 'message': 'Wajah tidak dikenali. Silakan daftar terlebih dahulu.'}), 404

    except Exception as e:
        print(f"ERROR saat face_login: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Terjadi kesalahan internal saat memproses gambar.'}), 500


def save_face_image(user_id: str, image_file) -> str:
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = str(int(time.time() * 1000))
    dst = os.path.join(user_dir, f"{timestamp}.jpg")
    try:
        image_file.save(dst)
        return dst
    except Exception as e:
        print(f"ERROR: Gagal menyimpan gambar ke {dst}: {e}")
        traceback.print_exc()
        return None

@app.route('/register_face', methods=['POST'])
def register_face():
    print("===== MULAI register_face =====")
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400
    
    raw_path = save_face_image(user_id, image)
    if raw_path is None:
        return jsonify({'success': False, 'error': 'Gagal menyimpan gambar yang diunggah'}), 500
    
    try:
        firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Gagal mengupload gambar ke Firebase'}), 500
    
    success = update_lbph_model_incrementally(raw_path, user_id)
    
    if success:
        print("INFO: Model diupdate, memuat ulang model global...")
        load_models_globally()

    try:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    except Exception as e:
        print(f"WARNING: Gagal menghapus file sementara {raw_path}: {e}")

    if not success:
        return jsonify({'success': False, 'error': 'Gagal mengupdate model LBPH'}), 500
    
    return jsonify({'success': True, 'firebase_image_url': firebase_url})


@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    
    # Proses gambar di memori
    try:
        in_memory_file = np.fromstring(image.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if global_recognizer is None or not global_labels_reverse:
            return jsonify({'success': False, 'error': 'Model belum ada atau label map kosong.'}), 400
            
        label, conf = global_recognizer.predict(gray)
        print(f"DEBUG: Predicted label: {label}, Confidence: {conf}")

        if label == -1 or conf > 90:
            return jsonify({'success': False, 'error': 'Wajah tidak dikenali.'}), 404
            
        recognized_user_id = global_labels_reverse.get(label)
        if recognized_user_id is None:
            return jsonify({'success': False, 'error': 'Label terprediksi tidak ditemukan di label map.'}), 500
            
        return jsonify({ 'success': True, 'user_id': recognized_user_id, 'confidence': float(conf) })
    except Exception as e:
        print(f"ERROR saat verify_face: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Terjadi error internal: {e}'}), 500


@app.route('/list_user_faces', methods=['GET'])
def list_user_faces():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Parameter user_id wajib diisi'}), 400
    user_dir = os.path.join(FACES_DIR, user_id)
    if not os.path.exists(user_dir):
        return jsonify({'success': True, 'files': []}), 200
    files = [f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')]
    return jsonify({'success': True, 'files': files}), 200

@app.route('/get_face_image', methods=['GET'])
def get_face_image():
    user_id = request.args.get('user_id')
    filename = request.args.get('filename')
    if not user_id or not filename:
        return jsonify({'success': False, 'error': 'Parameter user_id dan filename wajib diisi'}), 400
    user_dir = os.path.join(FACES_DIR, user_id)
    if not os.path.exists(os.path.join(user_dir, filename)):
        return jsonify({'success': False, 'error': 'File tidak ditemukan'}), 404
    return send_from_directory(user_dir, filename)

# ... (endpoint find_my_photos dan debug_ls bisa ditambahkan kembali jika masih diperlukan) ...

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    load_models_globally()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
else:
    # Ini akan dipanggil ketika dijalankan oleh Gunicorn di production
    load_models_globally()