import os
import json
import traceback

import cv2
import numpy as np # WAJIB ditambahkan untuk memproses gambar
from flask import Flask, request, jsonify, send_from_directory

# ----------------- PERUBAHAN DI SINI -----------------
# Hapus import yang lama dan tidak akurat
# from face_preprocessing import detect_and_crop 

# GANTI dengan import fungsi deteksi wajah yang lebih kuat dari gdrive_match
from gdrive_match import detect_and_crop_face as detect_and_crop
# ----------------- AKHIR PERUBAHAN -----------------

from face_data import train_and_evaluate
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage

from gdrive_match import find_all_matching_photos, get_all_gdrive_folder_ids

if not firebase_admin._apps:
    import os, json
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

# Fungsi helper untuk upload file ke Firebase Storage
def upload_to_firebase(local_file, user_id, filename):
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
    return blob.public_url

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exceptions(e):
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

FACES_DIR       = "faces"
MODEL_PATH      = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"

os.makedirs(FACES_DIR, exist_ok=True)

def save_face_image(user_id: str, image_file_path: str):
    """
    Simpan gambar yang SUDAH DICROP ke folder faces/<user_id>/N.jpg
    """
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')])
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    # Pindahkan file yang sudah di-crop ke tujuan
    os.rename(image_file_path, dst)
    return dst
    
def load_model_and_labels():
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

@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

# ----------------- PERUBAHAN UTAMA DI FUNGSI INI -----------------
@app.route('/register_face', methods=['POST'])
def register_face():
    print("===== MULAI register_face (versi baru) =====")
    user_id = request.form.get('user_id')
    image_file = request.files.get('image')

    if not user_id or not image_file:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400

    # 1. Baca gambar yang di-upload ke memori
    in_memory_file = image_file.read()
    np_arr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'success': False, 'error': 'Gagal membaca file gambar'}), 400

    # 2. Gunakan fungsi deteksi MTCNN yang kuat
    cropped = detect_and_crop(img)
    
    if cropped is None or cropped.size == 0:
        print("Gagal cropping dengan MTCNN!")
        return jsonify({'success': False, 'error': 'Gagal cropping/gambar kosong!'}), 400

    # 3. Simpan gambar yang SUDAH di-crop ke file sementara
    temp_path = "temp_cropped.jpg"
    cv2.imwrite(temp_path, cropped)

    # 4. Pindahkan file sementara ke direktori user
    final_path = save_face_image(user_id, temp_path)
    
    # 5. Upload ke Firebase Storage
    firebase_url = upload_to_firebase(final_path, user_id, os.path.basename(final_path))

    # 6. Retrain model
    metrics = train_and_evaluate()
    return jsonify({ 'success': True, 'metrics': metrics, 'firebase_image_url': firebase_url })
# ----------------- AKHIR PERUBAHAN UTAMA -----------------


@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    
    in_memory_file = image.read()
    np_arr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'success': False, 'error': 'Gagal membaca file gambar'}), 400

    gray = detect_and_crop(img) # Gunakan detektor yang sudah diperbaiki
    if gray is None:
         return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400
    
    # Convert ke grayscale untuk LBPH
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    model, lblmap = load_model_and_labels()
    if model is None:
        return jsonify({'success': False, 'error': 'Model belum ada'}), 400

    label, conf = model.predict(gray)
    return jsonify({ 'success': True, 'user_id': lblmap[label], 'confidence': float(conf) })


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
        image = request.files['image']
        user_tmp = 'tmp_user.jpg'
        image.save(user_tmp)

        lbph_model = cv2.face.LBPHFaceRecognizer_create()
        lbph_model.read('lbph_model.xml')

        all_folder_ids = get_all_gdrive_folder_ids()
        matches = find_all_matching_photos(user_tmp, all_folder_ids, lbph_model, threshold=70)

        print("RESPONSE:", matches)

        return jsonify({'success': True, 'matched_photos': matches})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# (Endpoint debug lainnya tidak perlu diubah)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))