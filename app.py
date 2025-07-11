# app.py (Final Dispatch Job)

import os
import json
import traceback
import numpy as np
import time # Diperlukan untuk timestamp

import cv2
from flask import Flask, request, jsonify, send_from_directory

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate # Untuk melatih model LBPH

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage, firestore # Tambahkan firestore di sini
from uuid import uuid4 # Untuk membuat ID tugas yang unik

if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

db = firestore.client() # Inisialisasi Firestore client

# Fungsi helper untuk upload file ke Firebase Storage
def upload_to_firebase(local_file, user_id, filename):
    """Upload file ke Firebase Storage dan return URL download-nya"""
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    
    # Upload dengan kualitas JPEG 90%
    # Pastikan local_file adalah path ke gambar yang sudah di-crop dan bersih
    blob.upload_from_filename(
        local_file,
        content_type='image/jpeg', # Atau 'image/png' tergantung format
        predefined_acl='publicRead'
    )
    
    return blob.public_url

app = Flask(__name__)

# ─── Error Handler ─────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exceptions(e):
    """Tangkap semua exception, print traceback, dan kembalikan JSON error."""
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

# ─── Konstanta ─────────────────────────────────────────────────────────────
FACES_DIR       = "faces"
MODEL_PATH      = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

# Pastikan direktori dataset ada
os.makedirs(FACES_DIR, exist_ok=True)

# ─── Helper Functions ──────────────────────────────────────────────────────
def save_face_image(user_id: str, image_file) -> str:
    """
    Simpan gambar upload ke folder faces/<user_id>/N.jpg
    Folder user_id akan otomatis dibuat kalau belum ada.
    Mengembalikan path file yang disimpan.
    """
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)  # Otomatis bikin folder user baru
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) # Menghitung semua ekstensi gambar
    dst = os.path.join(user_dir, f"{count+1}.jpg") # Selalu simpan sebagai JPG
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

    # Simpan gambar mentah yang diupload
    raw_path = save_face_image(user_id, image)
    
    # Baca gambar mentah untuk diproses
    img_raw = cv2.imread(raw_path)
    if img_raw is None:
        print(f"Gagal memuat gambar dari {raw_path}")
        return jsonify({'success': False, 'error': 'Gagal memuat gambar yang diupload.'}), 400

    # Deteksi dan crop wajah. Ini akan mengembalikan NumPy array grayscale.
    cropped_face_np = detect_and_crop(img_raw)
    
    if cropped_face_np is None or cropped_face_np.size == 0:
        print("Gagal cropping/tidak ada wajah terdeteksi!")
        return jsonify({'success': False, 'error': 'Gagal mendeteksi atau memotong wajah dari gambar.'}), 400
    
    # Timpa gambar asli dengan gambar wajah yang sudah di-crop dan grayscale
    cv2.imwrite(raw_path, cropped_face_np)
    
    # Upload ke Firebase Storage
    firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))

    # Retrain dan simpan model
    # Pastikan train_and_evaluate() mengambil data dari FACES_DIR dan menghasilkan MODEL_PATH & LABELS_MAP_PATH
    metrics = train_and_evaluate() 
    
    return jsonify({ 'success': True, 'metrics': metrics, 'firebase_image_url': firebase_url })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    
    # Simpan gambar sementara
    tmp_path = 'tmp_verify.jpg'
    image.save(tmp_path)

    # Baca gambar sementara
    img_raw = cv2.imread(tmp_path)
    if img_raw is None:
        os.remove(tmp_path)
        return jsonify({'success': False, 'error': 'Gagal memuat gambar untuk verifikasi.'}), 400

    # Deteksi dan crop wajah
    cropped_face_np = detect_and_crop(img_raw)
    os.remove(tmp_path) # Hapus file sementara setelah diproses

    if cropped_face_np is None or cropped_face_np.size == 0:
        return jsonify({'success': False, 'error': 'Tidak ada wajah terdeteksi untuk verifikasi.'}), 400

    model, lblmap = load_model_and_labels()
    if model is None:
        return jsonify({'success': False, 'error': 'Model pengenalan wajah belum dilatih.'}), 400
    
    # Prediksi menggunakan model LBPH
    label, conf = model.predict(cropped_face_np)
    
    # Pastikan label ada di label_map
    user_id_predicted = lblmap.get(label, "unknown")

    # Anda bisa menambahkan threshold di sini juga untuk verifikasi
    # Misalnya, jika confidence terlalu tinggi (tidak mirip), anggap tidak cocok
    # if conf > YOUR_VERIFICATION_THRESHOLD:
    #     user_id_predicted = "unknown"

    return jsonify({ 'success': True, 'user_id': user_id_predicted, 'confidence': float(conf) })

# --- Endpoint untuk memulai pencarian foto (DISPATCH JOB) ---
@app.route('/start_photo_search', methods=['POST'])
def start_photo_search():
    print("\n===== Endpoint /start_photo_search DIPANGGIL (DISPATCHING JOB) =====\n")
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'File gambar tidak ditemukan'}), 400

    image = request.files['image']
    job_id = str(uuid4()) # Buat ID tugas unik
    
    # Simpan gambar klien ke Firebase Storage untuk diakses oleh worker
    # Gunakan nama file yang unik untuk job ini
    client_image_filename = f"client_search_face_{job_id}.jpg"
    temp_local_path = os.path.join("/tmp", client_image_filename) # Simpan sementara di /tmp
    image.save(temp_local_path)

    # Baca gambar yang baru disimpan untuk diproses oleh detect_and_crop
    img_raw_client = cv2.imread(temp_local_path)
    if img_raw_client is None:
        os.remove(temp_local_path)
        return jsonify({'success': False, 'error': 'Gagal memuat gambar klien untuk pencarian.'}), 400

    # Deteksi dan crop wajah klien
    cropped_client_face_np = detect_and_crop(img_raw_client)
    
    if cropped_client_face_np is None or cropped_client_face_np.size == 0:
        os.remove(temp_local_path)
        return jsonify({'success': False, 'error': 'Tidak ada wajah terdeteksi pada gambar klien untuk pencarian.'}), 400

    # Timpa file sementara dengan wajah yang sudah di-crop dan grayscale
    cv2.imwrite(temp_local_path, cropped_client_face_np)

    # Upload gambar wajah klien yang sudah di-crop ke Firebase Storage
    # Ini akan diakses oleh worker
    client_image_url = upload_to_firebase(temp_local_path, "search_clients", client_image_filename)
    os.remove(temp_local_path) # Hapus file lokal sementara

    # Buat entri tugas di Firestore
    job_data = {
        'clientImageURL': client_image_url,
        'status': 'pending', # Status awal
        'progress': 0,
        'total': 0, # Akan diisi oleh worker
        'results': [],
        'error': None,
        'createdAt': firestore.SERVER_TIMESTAMP # Timestamp kapan job dibuat
    }
    db.collection('photo_search_jobs').document(job_id).set(job_data)

    print(f"Tugas pencarian dibuat dengan job_id: {job_id}")
    return jsonify({'success': True, 'job_id': job_id})

# --- Endpoint untuk mendapatkan status pencarian foto ---
@app.route('/get_search_status', methods=['GET'])
def get_search_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'success': False, 'error': 'Parameter job_id wajib diisi'}), 400
        
    job_ref = db.collection('photo_search_jobs').document(job_id)
    job_doc = job_ref.get()

    if not job_doc.exists:
        return jsonify({'success': False, 'error': 'Tugas tidak ditemukan'}), 404

    return jsonify({'success': True, 'job_data': job_doc.to_dict()})


# --- Debug Endpoint: Lihat Isi Folder Railway ---
@app.route('/debug_ls', methods=['GET'])
def debug_ls():
    result = {}
    for folder in ['faces', '.', 'models', '/tmp']: # Tambahkan /tmp untuk debugging
        try:
            result[folder] = os.listdir(folder)
        except Exception as e:
            result[folder] = str(e)
    return jsonify(result)

@app.route('/debug_ls2')
def debug_ls2():
    return jsonify({
        '.': os.listdir('.'),
        'faces/user123': os.listdir('faces/user123'),
        'faces/user123_test': os.listdir('faces/user123_test'),
    })

@app.route('/debug_ls3')
def debug_ls3():
    return jsonify({
        'root': os.listdir('.'),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

