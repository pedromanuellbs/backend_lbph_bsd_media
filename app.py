import os
import json
import traceback

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

# Import yang masih relevan
from face_preprocessing import detect_and_crop
from config import FACES_DIR, MODEL_PATH, LABEL_MAP
from gdrive_match import find_all_matching_photos, get_all_gdrive_folder_ids

# --- Import dan setup Firebase Admin SDK (masih digunakan oleh gdrive_match) ---
import firebase_admin
from firebase_admin import credentials

if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred) # Inisialisasi dasar sudah cukup

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

# ROUTE /register_face SUDAH DIHAPUS

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

        model, lblmap = load_model_and_labels()
        if model is None:
            return jsonify({'success': False, 'error': 'Model pencocokan belum siap.'}), 503


        all_folder_ids = get_all_gdrive_folder_ids()
        matches = find_all_matching_photos(user_tmp, all_folder_ids, model, threshold=70)

        # Hapus file temp setelah selesai
        os.remove(user_tmp)

        print("RESPONSE:", matches)
        return jsonify({'success': True, 'matched_photos': matches})
    except Exception as e:
        print("===== ERROR TRACEBACK =====")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))