import os
import json
import traceback
import numpy as np
import time

import cv2
from flask import Flask, request, jsonify, send_from_directory

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate

from config import FACES_DIR, MODEL_PATH, LABEL_MAP
from uuid import uuid4 # Untuk membuat ID tugas yang unik

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage, firestore # <--- TAMBAHKAN firestore di sini

if not firebase_admin._apps:
    import os, json
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })




# Fungsi helper untuk upload file ke Firebase Storage
def upload_to_firebase(local_file, user_id, filename):
    bucket = storage.bucket('db-ta-bsd-media.firebasestorage.app')
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    
    # Upload dengan kualitas JPEG 90%
    blob.upload_from_filename(
        local_file,
        content_type='image/jpeg',
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
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')])
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(dst)
    return dst
#anjay
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
    start_time = time.time()
    
    # Validasi input
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau file gambar tidak valid/kosong'}), 400
    
    # Validasi ekstensi file
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    if not any(image.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({'success': False, 'error': 'Format gambar tidak didukung. Gunakan JPG, JPEG, atau PNG'}), 400
    
    # Validasi ukuran file (maksimal 5 MB)
    try:
        # Simpan sementara untuk cek ukuran
        temp_path = f"/tmp/{uuid4().hex}.tmp"
        image.save(temp_path)
        file_size = os.path.getsize(temp_path)
        os.remove(temp_path)
        
        if file_size > 5 * 1024 * 1024:  # 5MB
            return jsonify({'success': False, 'error': 'Ukuran gambar terlalu besar (maksimal 5MB)'}), 400
    except Exception as e:
        print(f"Error validasi ukuran file: {e}")
        return jsonify({'success': False, 'error': 'Gagal memvalidasi ukuran file'}), 400
    
    # Simpan gambar asli ke folder user
    try:
        raw_path = save_face_image(user_id, image)
        print(f"Gambar disimpan sementara di: {raw_path}")
    except Exception as e:
        print(f"Error menyimpan gambar: {e}")
        return jsonify({'success': False, 'error': f'Gagal menyimpan file: {e}'}), 500

    # Proses deteksi dan crop wajah
    try:
        # Panggil fungsi detect_and_crop
        cropped = detect_and_crop(raw_path)
    except Exception as e:
        print(f"Error selama deteksi wajah: {e}")
        # Hapus file yang gagal
        try:
            os.remove(raw_path)
        except:
            pass
        return jsonify({'success': False, 'error': f'Gagal memproses wajah: {e}'}), 500

    # Cek apakah wajah terdeteksi
    if cropped is None:
        print(f"Wajah tidak terdeteksi untuk user: {user_id}")
        try:
            os.remove(raw_path)
        except:
            pass
        return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi. Pastikan wajah terlihat jelas dan tidak tertutup.'}), 400

    # Timpa file asli dengan hasil crop (dalam format grayscale)
    cv2.imwrite(raw_path, cropped)
    print(f"Wajah berhasil di-crop dan disimpan ulang di: {raw_path}")

    # Upload ke Firebase Storage
    try:
        firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))
        print(f"Gambar diupload ke Firebase: {firebase_url}")
    except Exception as e:
        print(f"Error upload Firebase: {e}")
        return jsonify({'success': False, 'error': f'Gagal mengunggah ke Firebase: {e}'}), 500

    # Latih ulang model dengan menambahkan data baru (opsional, bisa di-comment jika tidak ingin langsung train)
    # try:
    #     print("Memulai pelatihan model...")
    #     metrics = train_and_evaluate()
    #     print(f"Model dilatih ulang. Metrics: {metrics}")
    # except Exception as e:
    #     print(f"Error selama pelatihan model: {e}")
        # Tidak mengembalikan error ke klien, karena registrasi sudah berhasil
        # Tapi log error untuk debugging

    elapsed = time.time() - start_time
    print(f"Registrasi wajah selesai dalam {elapsed:.2f} detik")
    
    return jsonify({
        'success': True,
        'firebase_image_url': firebase_url,
        'processing_time': elapsed
    })

def save_face_image(user_id: str, image_file) -> str:
    """
    Simpan gambar upload ke folder faces/<user_id>/N.jpg
    Folder user_id akan otomatis dibuat kalau belum ada.
    Mengembalikan path file yang disimpan.
    """
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # Hitung file yang sudah ada
    existing_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    next_num = len(existing_files) + 1
    
    # Buat nama file baru
    dst = os.path.join(user_dir, f"{next_num}.jpg")
    
    # Simpan file
    image_file.save(dst)
    
    return dst


@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    # (Opsional tapi direkomendasikan) Tambahkan secret key untuk keamanan
    secret_key = request.headers.get('X-Secret-Key')
    if secret_key != "KATA_RAHASIA_ANDA": # Ganti dengan secret key yang aman
        return jsonify({'success': False, 'error': 'Akses ditolak'}), 403

    try:
        print("Memulai pelatihan model secara manual...")
        start_time = time.time()
        metrics = train_and_evaluate()
        elapsed = time.time() - start_time
        print(f"Model berhasil dilatih ulang dalam {elapsed:.2f} detik. Metrics: {metrics}")
        
        return jsonify({
            'success': True, 
            'message': 'Model berhasil dilatih ulang.',
            'metrics': metrics,
            'training_time': elapsed
        })
    except Exception as e:
        print(f"Error selama pelatihan manual: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    # preprocess & predict
    tmp = 'tmp.jpg'; image.save(tmp)
    gray = detect_and_crop(tmp); os.remove(tmp)
    model, lblmap = load_model_and_labels()
    if model is None:
        return jsonify({'success': False, 'error': 'Model belum ada'}), 400
    label, conf = model.predict(gray)
    return jsonify({ 'success': True, 'user_id': lblmap[label], 'confidence': float(conf) })

# --- Tambahan: Endpoint untuk melihat daftar file wajah user ---
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

# --- Tambahan: Endpoint untuk download/lihat gambar user tertentu ---
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

import traceback

# Di file: app.py

# GANTI endpoint /find_my_photos
@app.route('/start_photo_search', methods=['POST'])
def start_photo_search():
    print("\n===== Endpoint /start_photo_search DIPANGGIL =====\n")
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'File gambar tidak ditemukan'}), 400

    image = request.files['image']
    
    # Buat ID unik untuk tugas ini
    job_id = str(uuid4())
    
    # Simpan gambar klien ke Firebase Storage agar bisa diakses worker
    # Anda sudah punya fungsi upload_to_firebase, kita pakai itu
    # Misal user_id untuk file sementara adalah 'search-requests'
    filename = f"{job_id}.jpg"
    tmp_path = f"/tmp/{filename}" # Gunakan folder /tmp yang biasanya tersedia
    image.save(tmp_path)
    
    client_image_url = upload_to_firebase(tmp_path, "search-requests", filename)
    os.remove(tmp_path)

    # Buat dokumen tugas baru di Firestore
    db = firestore.client()
    job_ref = db.collection('photo_search_jobs').document(job_id)
    job_ref.set({
        'status': 'pending',
        'createdAt': firestore.SERVER_TIMESTAMP,
        'clientImageURL': client_image_url,
        'results': [],
        'error': None
    })

    print(f"Tugas pencarian dibuat dengan job_id: {job_id}")
    return jsonify({'success': True, 'job_id': job_id})

# TAMBAHKAN endpoint baru ini
@app.route('/get_search_status', methods=['GET'])
def get_search_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'success': False, 'error': 'Parameter job_id wajib diisi'}), 400
        
    db = firestore.client()
    job_ref = db.collection('photo_search_jobs').document(job_id)
    job_doc = job_ref.get()

    if not job_doc.exists:
        return jsonify({'success': False, 'error': 'Tugas tidak ditemukan'}), 404

    return jsonify({'success': True, 'job_data': job_doc.to_dict()})





# --- Debug Endpoint: Lihat Isi Folder Railway ---
@app.route('/debug_ls', methods=['GET'])
def debug_ls():
    result = {}
    for folder in ['faces', '.', 'models']:
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
