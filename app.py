import os
import traceback

import cv2
from flask import Flask, request, jsonify

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate

from config import FACES_DIR, MODEL_PATH, LABEL_MAP

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
    # retrain dan simpan model
    metrics = train_and_evaluate()
    return jsonify({ 'success': True, 'metrics': metrics })


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)