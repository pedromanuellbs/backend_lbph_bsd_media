import os
import traceback
import json
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage, firestore

# --- Konstanta ---
FIREBASE_BUCKET_NAME = "db-ta-bsd-media.firebasestorage.app"
DRIVE_API_KEY = "AIzaSyC_vPd6yPwYQ60Pn-tuR3Nly_7mgXZcxGk" # Diambil dari kode Flutter

# --- Load credential dari environment variable ---
cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
cred = credentials.Certificate(cred_info)
firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET_NAME})

# --- Inisialisasi Service Firebase ---
bucket = storage.bucket()
db = firestore.client()

app = Flask(__name__)

# ─── Error Handler ─────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exceptions(e):
    """Tangkap semua exception, print traceback, dan kembalikan JSON error."""
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

# ─── Konstanta Lokal ───────────────────────────────────────────────────────
# Variabel dari config.py mungkin sudah diimpor, tapi kita definisikan lagi
# untuk kejelasan jika file config tidak ada.
FACES_DIR       = "faces"
MODEL_PATH      = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

# Pastikan direktori dataset ada
os.makedirs(FACES_DIR, exist_ok=True)

# ─── Helper Functions ──────────────────────────────────────────────────────
def upload_to_firebase(local_file, user_id, filename):
    """Upload file ke Firebase Storage dan return URL download-nya"""
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
    return blob.public_url

def save_face_image(user_id: str, image_file) -> str:
    """
    Simpan gambar upload ke folder faces/<user_id>/N.jpg
    Folder user_id akan otomatis dibuat kalau belum ada.
    Mengembalikan path file yang disimpan.
    """
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')])
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(dst)
    return dst

def load_model_and_labels():
    """
    Load model LBPH dan label_map (lbl→user_id) dari filesystem.
    Jika belum ada model, kembalikan (None, {}).
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_MAP_PATH):
        return None, {}

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)

    label_map = {}
    with open(LABELS_MAP_PATH, "r") as f:
        for line in f:
            lbl, uid = line.strip().split(":")
            label_map[int(lbl)] = uid

    return model, label_map

def fetch_images_from_drive_folder(folder_url):
    """Mengambil daftar URL gambar dari sebuah folder Google Drive."""
    urls = []
    folder_id = None
    if '/d/' in folder_url:
        folder_id = folder_url.split('/d/')[1].split('/')[0]
    elif '/folders/' in folder_url:
        folder_id = folder_url.split('/folders/')[1].split('/')[0]
    
    if not folder_id:
        return []

    api_url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': f"'{folder_id}' in parents and mimeType contains 'image/'",
        'fields': 'files(id)',
        'key': DRIVE_API_KEY,
        'pageSize': 1000
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        for file_info in data.get('files', []):
            download_url = f"https://drive.google.com/uc?export=download&id={file_info['id']}"
            urls.append(download_url)
    except requests.exceptions.RequestException as e:
        print(f"Error mengakses Google Drive API: {e}")

    return urls

# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

@app.route('/register_face', methods=['POST'])
def register_face():
    print("===== MULAI register_face =====")
    user_id = request.form.get('user_id')
    image   = request.files.get('image')

    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400

    raw_path = save_face_image(user_id, image)
    cropped  = detect_and_crop(raw_path)

    if cropped is None or cropped.size == 0:
        return jsonify({'success': False, 'error': 'Gagal cropping/wajah tidak terdeteksi'}), 400
    cv2.imwrite(raw_path, cropped)
    
    firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))

    metrics = train_and_evaluate()
    return jsonify({ 'success': True, 'metrics': metrics, 'firebase_image_url': firebase_url })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400

    tmp = 'tmp.jpg'; image.save(tmp)
    gray = detect_and_crop(tmp); os.remove(tmp)
    
    if gray is None:
        return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400

    model, lblmap = load_model_and_labels()
    if model is None:
        return jsonify({'success': False, 'error': 'Model belum ada'}), 400

    label, conf = model.predict(gray)
    return jsonify({ 'success': True, 'user_id': lblmap.get(label, 'unknown'), 'confidence': float(conf) })

@app.route('/find_my_photos', methods=['POST'])
def find_my_photos():
    print("===== MULAI find_my_photos =====")
    image = request.files.get('image')

    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400

    tmp_path = 'tmp_find.jpg'
    image.save(tmp_path)
    client_face_gray = detect_and_crop(tmp_path)
    os.remove(tmp_path)

    if client_face_gray is None:
        return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi pada gambar yang di-upload'}), 400

    model, label_map = load_model_and_labels()
    if model is None:
        return jsonify({'success': False, 'error': 'Model belum ada, tidak bisa melakukan pencocokan'}), 400

    label, confidence = model.predict(client_face_gray)
    client_user_id = label_map.get(label, 'unknown')
    print(f"Wajah terverifikasi sebagai user_id: {client_user_id} dengan confidence: {confidence}")

    if client_user_id == 'unknown':
        return jsonify({'success': True, 'user_id': 'unknown', 'photo_urls': []})

    matching_urls = []
    sessions_ref = db.collection('photo_sessions').stream()
    
    for session in sessions_ref:
        session_data = session.to_dict()
        drive_link = session_data.get('driveLink')
        if not drive_link:
            continue
        
        print(f"Memeriksa sesi: {session_data.get('title')}...")
        photo_urls_in_drive = fetch_images_from_drive_folder(drive_link)

        for photo_url in photo_urls_in_drive:
            try:
                response = requests.get(photo_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open("temp_drive_photo.jpg", 'wb') as f:
                        f.write(response.content)

                    drive_photo_gray = detect_and_crop("temp_drive_photo.jpg")
                    if drive_photo_gray is not None:
                        predicted_label, _ = model.predict(drive_photo_gray)
                        predicted_user_id = label_map.get(predicted_label)
                        
                        if predicted_user_id == client_user_id:
                            print(f"COCOK! Foto {photo_url} adalah milik {client_user_id}")
                            matching_urls.append(photo_url)
            except Exception as e:
                print(f"Error memproses foto dari drive {photo_url}: {e}")
        
        if os.path.exists("temp_drive_photo.jpg"):
            os.remove("temp_drive_photo.jpg")

    print(f"Ditemukan {len(matching_urls)} foto yang cocok.")
    return jsonify({
        'success': True,
        'user_id': client_user_id,
        'photo_urls': matching_urls
    })

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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)