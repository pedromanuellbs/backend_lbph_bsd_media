import os
import json
import traceback

import numpy as np  # added for dtype conversion
import cv2
from flask import Flask, request, jsonify, send_from_directory

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate

from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# --- Import dan setup Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage

from gdrive_match import find_matching_photos, find_all_matching_photos, get_all_gdrive_folder_ids

if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

# Fungsi helper untuk upload file ke Firebase Storage
def upload_to_firebase(local_file, user_id, filename):
    """Upload file ke Firebase Storage dan return URL download-nya"""
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
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

os.makedirs(FACES_DIR, exist_ok=True)

# ─── Helper Functions ──────────────────────────────────────────────────────
def save_face_image(user_id: str, image_file) -> str:
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len([f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')])
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(dst)
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

# ─── Routes ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

@app.route('/register_face', methods=['POST'])
def register_face():
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

    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400

    raw_path = save_face_image(user_id, image)
    cropped  = detect_and_crop(raw_path)
    if cropped is None or cropped.size == 0:
        print("Gagal cropping/gambar kosong!")
        return jsonify({'success': False, 'error': 'Gagal cropping/gambar kosong!'}), 400

    # jika perlu, convert array dtype sebelum penyimpanan atau pengolahan lebih lanjut
    # cropped = cropped.astype(np.float32)
    cv2.imwrite(raw_path, cropped)

    # Upload ke Firebase Storage
    firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))

    # retrain dan simpan model
    metrics = train_and_evaluate()
    # Cast numpy types to Python native types untuk JSON encoding
    metrics = {
        k: (int(v) if isinstance(v, np.generic) else v)
        for k, v in metrics.items()
    }

    return jsonify({
        'success': True,
        'metrics': metrics,
        'firebase_image_url': firebase_url
    })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    tmp = 'tmp.jpg'; image.save(tmp)
    gray = detect_and_crop(tmp); os.remove(tmp)
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
        print("===== ERROR TRACEBACK =====")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
    return jsonify({ 'root': os.listdir('.') })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
