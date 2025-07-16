print("=== DEBUG: Ini app.py terbaru versi 2025-07-17 (SUDAH DIPERBAIKI) ===")

import os
import sys
import traceback
import json
import tempfile
import numpy as np
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory
import time
import firebase_admin
from firebase_admin import credentials, storage

from face_preprocessing import detect_and_crop
from face_data import update_lbph_model_incrementally, load_model_and_labels
from config import FACES_DIR, MODEL_PATH, LABEL_MAP
from gdrive_match import find_matching_photos, find_all_matching_photos, get_all_gdrive_folder_ids

# --- Global variables for models ---
global_recognizer = None
global_labels_reverse = None

# --- Firebase Admin SDK Initialization ---
try:
    if not firebase_admin._apps:
        cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        cred = credentials.Certificate(cred_info)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
        })
        print("Firebase Admin SDK initialized successfully.")
except KeyError:
    print("ERROR: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON not set.")
except Exception as e:
    print(f"ERROR initializing Firebase Admin SDK: {e}")

bucket = storage.bucket()

# --- Helper Functions ---
def upload_to_firebase(local_file, user_id, filename):
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
    except Exception as e:
        print(f"ERROR: Gagal mengunduh file dari URL {url}: {e}")
        return False

# --- ✨ NEW: Function to load models globally at startup ---
def load_models_globally():
    global global_recognizer, global_labels_reverse
    print("INFO: Memeriksa dan memuat model LBPH ke memori...")

    MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flbph_model.xml?alt=media&token=26656ed8-3cd1-4220-a07d-aad9aaeb91f5"
    LABEL_MAP_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flabels_map.txt?alt=media&token=2ab5957f-78b2-41b0-a1aa-b2f1b8675f54"

    # Ensure local directory exists
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"INFO: {MODEL_PATH} tidak ditemukan. Mengunduh...")
        if not download_file_from_url(MODEL_URL, MODEL_PATH):
            print("CRITICAL: Gagal mengunduh model. Fungsi login tidak akan bekerja.")
            return
    if not os.path.exists(LABEL_MAP):
        print(f"INFO: {LABEL_MAP} tidak ditemukan. Mengunduh...")
        if not download_file_from_url(LABEL_MAP_URL, LABEL_MAP):
            print("CRITICAL: Gagal mengunduh label map. Fungsi login tidak akan bekerja.")
            return

    try:
        global_recognizer = cv2.face.LBPHFaceRecognizer_create()
        global_recognizer.read(MODEL_PATH)

        labels = {}
        with open(LABEL_MAP, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ":" in line:
                    # ✨ PERBAIKAN KUNCI ADA DI SINI
                    k, v = line.split(":", 1) # k adalah id_angka, v adalah nama
                    # Simpan dengan format: { 'nama_user': id_angka }
                    labels[v.strip()] = int(k.strip())
        
        # Buat pemetaan sebaliknya: { id_angka: 'nama_user' }
        global_labels_reverse = {v: k for k, v in labels.items()}
        print("INFO: Model LBPH dan label map berhasil dimuat ke memori global.")
    except Exception as e:
        print(f"CRITICAL: Gagal memuat model atau label dari file: {e}")
        traceback.print_exc()


app = Flask(__name__)

# --- Error Handler ---
@app.errorhandler(Exception)
def handle_exceptions(e):
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

# --- ✨ MODIFIED: `/face_login` endpoint ---
@app.route('/face_login', methods=['POST'])
def face_login():
    print("REQUEST MASUK KE /face_login (LOGIKA BARU)")
    
    # Ambil username dan gambar dari request
    claimed_username = request.form.get('user_id')
    image_file = request.files.get('image')

    # Validasi input
    if not claimed_username or not image_file:
        return jsonify({'error': 'user_id dan image wajib diisi'}), 400

    # Pastikan model sudah siap di memori
    if global_recognizer is None or global_labels_reverse is None:
        print("ERROR: Model tidak siap. Login tidak dapat diproses.")
        return jsonify({'error': 'Server model is not ready, please try again later.'}), 503

    try:
        # Baca gambar langsung dari memori
        in_memory_file = np.fromstring(image_file.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Format gambar tidak valid atau korup.'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) == 0:
            return jsonify({'status': 'not_found', 'message': 'Wajah tidak terdeteksi di foto.'}), 400

        # Hanya proses wajah pertama yang terdeteksi untuk login
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        # Prediksi wajah menggunakan model
        predicted_id, confidence = global_recognizer.predict(roi_gray)

        print(f"DEBUG: Claimed User: {claimed_username}, Predicted ID: {predicted_id}, Confidence: {confidence}")

        # Verifikasi hasil prediksi
        # Confidence < 60 dianggap cocok
        if confidence < 60 and predicted_id in global_labels_reverse:
            recognized_username = global_labels_reverse[predicted_id]
            print(f"DEBUG: Recognized User: {recognized_username}")

            # INI BAGIAN KUNCINYA: Bandingkan hasil prediksi dengan username yang dikirim
            if recognized_username == claimed_username:
                # Jika cocok, login berhasil
                return jsonify({
                    'status': 'success',
                    'message': 'Login berhasil!',
                    'user': {
                        'username': recognized_username,
                        'confidence': float(confidence)
                    }
                }), 200
            else:
                # Wajah dikenali sebagai orang lain
                return jsonify({'status': 'match_failed', 'message': 'Wajah Anda tidak cocok dengan username yang dimasukkan.'}), 403
        else:
            # Wajah tidak dikenali di database
            return jsonify({'status': 'not_found', 'message': 'Wajah tidak dikenali. Silakan daftar terlebih dahulu.'}), 404

    except Exception as e:
        print(f"ERROR saat face_login: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Terjadi kesalahan internal saat memproses gambar.'}), 500


# --- Other Endpoints (Unchanged) ---
os.makedirs(FACES_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

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
        return None

@app.route('/register_face', methods=['POST'])
def register_face():
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
    
    # After model is updated, we must reload it into memory
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
    tmp_filename = 'tmp_verify.jpg'
    os.makedirs(os.path.dirname(tmp_filename) or '.', exist_ok=True)
    try:
        image.save(tmp_filename)
        gray = detect_and_crop(tmp_filename)
        if gray is None:
            return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi atau gambar kosong'}), 400
        model, lblmap = load_model_and_labels()
        if model is None or not lblmap:
            return jsonify({'success': False, 'error': 'Model belum ada atau label map kosong.'}), 400
        label, conf = model.predict(gray)
        if label == -1 or conf > 90:
            return jsonify({'success': False, 'error': 'Wajah tidak dikenali.'}), 404
        recognized_user_id = lblmap.get(label)
        if recognized_user_id is None:
            return jsonify({'success': False, 'error': 'Label terprediksi tidak ditemukan di label map.'}), 500
        return jsonify({ 'success': True, 'user_id': recognized_user_id, 'confidence': float(conf) })
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


# (Keep other endpoints like list_user_faces, get_face_image, find_my_photos, debug_ls as they are)

@app.route('/list_user_faces', methods=['GET'])
def list_user_faces():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Parameter user_id wajib diisi'}), 400
    user_dir = os.path.join(FACES_DIR, user_id)
    if not os.path.exists(user_dir):
        return jsonify({'success': False, 'files': [], 'error': 'User belum punya data'}), 200
    files = [f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')]
    return jsonify({'success': True, 'files': files}), 200

@app.route('/get_face_image', methods=['GET'])
def get_face_image():
    user_id = request.args.get('user_id')
    filename = request.args.get('filename')
    if not user_id or not filename:
        return jsonify({'success': False, 'error': 'Parameter user_id dan filename wajib diisi'}), 400
    user_dir = os.path.join(FACES_DIR, user_id)
    file_path = os.path.join(user_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File tidak ditemukan'}), 404
    return send_from_directory(user_dir, filename)

@app.route('/find_my_photos', methods=['POST'])
def find_my_photos():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'File gambar tidak ditemukan'}), 400
        image = request.files['image']
        user_tmp = 'tmp_user_search.jpg'
        image.save(user_tmp)
        
        drive_links = request.form.get('drive_links')
        if drive_links:
            drive_folders = json.loads(drive_links)
            matched_photos = []
            folder_data = get_all_gdrive_folder_ids()
            folderid_to_docid = {fd['drive_folder_id']: fd['firestore_doc_id'] for fd in folder_data}
            for link in drive_folders:
                if 'folders/' in link:
                    folder_id = link.split('folders/')[1].split('?')[0]
                    session_id = folderid_to_docid.get(folder_id)
                    matches = find_matching_photos(user_tmp, folder_id, session_id)
                    for m in matches:
                        m['sessionId'] = session_id
                    matched_photos.extend(matches)
            return jsonify({'success': True, 'matched_photos': matched_photos})
        else:
            all_folder_data = get_all_gdrive_folder_ids()
            matches = find_all_matching_photos(user_tmp, all_folder_data)
            for m in matches:
                session_id = m.get('sessionId') or m.get('folder_id')
                m['sessionId'] = session_id
            return jsonify({'success': True, 'matched_photos': matches})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists('tmp_user_search.jpg'):
            os.remove('tmp_user_search.jpg')

@app.route('/debug_ls', methods=['GET'])
def debug_ls():
    result = {}
    try: result['root'] = os.listdir('.')
    except Exception as e: result['root'] = str(e)
    try: result['faces_dir'] = os.listdir(FACES_DIR)
    except Exception as e: result['faces_dir'] = str(e)
    user_id_param = request.args.get('user_id')
    if user_id_param:
        user_specific_dir = os.path.join(FACES_DIR, user_id_param)
        try: result[f'faces/{user_id_param}'] = os.listdir(user_specific_dir)
        except Exception as e: result[f'faces/{user_id_param}'] = str(e)
    result['model_path_exists'] = os.path.exists(MODEL_PATH)
    result['label_map_path_exists'] = os.path.exists(LABEL_MAP)
    return jsonify(result)

# --- Main Execution ---
if __name__ == '__main__':
    # Load models when the script starts
    load_models_globally()
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
else:
    # This block will run when deployed on a production server like Gunicorn
    load_models_globally()