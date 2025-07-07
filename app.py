import os
import json
import traceback
import glob

import cv2
from flask import Flask, request, jsonify, send_from_directory

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate
from gdrive_match import find_all_matching_photos_for_user
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage

from gdrive_match import find_matching_photos

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
    """Upload file ke Firebase Storage dan return URL download-nya"""
    bucket = storage.bucket()  # <--- Tambahin baris ini!
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()  # Atur permission sesuai kebutuhan
    return blob.public_url

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_dev_secret')

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
    # --- DEBUG LOGGING UNTUK TROUBLESHOOTING UPLOAD ---
    print("===== MULAI register_face =====")
    print("request.files:", request.files)
    print("request.form:", request.form)
    user_id = request.form.get('user_id')
    image   = request.files.get('image')
    if image:
        print("image.filename:", image.filename)
        print("image.content_length:", image.content_length)
    else:
        print("Tidak ada file 'image' di request!")

    # Validasi awal
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400

    # Simpan gambar ke folder user baru/eksisting (folder otomatis dibuat jika belum ada)
    raw_path = save_face_image(user_id, image)
    cropped  = detect_and_crop(raw_path)
    # --- Tambahkan pengecekan hasil crop ---
    if cropped is None or cropped.size == 0:
        print("Gagal cropping/gambar kosong!")
        return jsonify({'success': False, 'error': 'Gagal cropping/gambar kosong!'}), 400
    cv2.imwrite(raw_path, cropped)
    
    # --- Upload ke Firebase Storage ---
    firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))

    # retrain dan simpan model
    metrics = train_and_evaluate()
    return jsonify({ 'success': True, 'metrics': metrics, 'firebase_image_url': firebase_url })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400

    tmp = 'tmp_verify.jpg'; image.save(tmp)
    gray = detect_and_crop(tmp); os.remove(tmp)

    model, lblmap = load_model_and_labels()
    if model is None or lblmap == {}:
        return jsonify({'success': False, 'error': 'Model belum ada'}), 400

    labels_user = [lbl for lbl, uid in lblmap.items() if uid == user_id]
    if not labels_user:
        return jsonify({'success': False, 'error': 'User belum terdaftar'}), 400

    label, conf = model.predict(gray)
    if label in labels_user and conf < 70:
        session['verified_user_id'] = user_id
        return jsonify({'success': True, 'user_id': user_id, 'confidence': float(conf)})
    else:
        return jsonify({'success': False, 'error': 'Verifikasi gagal', 'confidence': float(conf)})


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

@app.route('/find_my_photos', methods=['POST'])
def find_my_photos():
    user_id = session.get('verified_user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Belum verifikasi wajah'}), 403
    try:
        lbph_model = cv2.face.LBPHFaceRecognizer_create()
        lbph_model.read('lbph_model.xml')
        all_folder_ids = get_all_gdrive_folder_ids()
        matches = find_all_matching_photos_for_user(user_id, all_folder_ids, lbph_model, threshold=70)
        return jsonify({'success': True, 'matched_photos': matches})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500




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
    app.run(host='0.0.0.0', port=8000)
# app.secret_key = 'random_secret_key_boleh_ganti'