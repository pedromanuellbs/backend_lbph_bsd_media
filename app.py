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
    Mengembalikan path file yang disimpan.
    """
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len(os.listdir(user_dir))
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


@app.route("/register_face", methods=["POST"])
def register_face():
    """
    Endpoint untuk registrasi muka:
    - Simpan gambar
    - Deteksi & crop→grayscale
    - Latih ulang model penuh
    - Kembalikan metrik evaluasi
    """
    user_id = request.form.get("user_id")
    image   = request.files.get("image")

    if not user_id or not image:
        return jsonify({'success': False, 'error': 'Missing user_id or image'}), 400

    # Simpan + preprocess
    raw_path = save_face_image(user_id, image)
    gray     = detect_and_crop(raw_path)
    if gray is None:
        return jsonify({'success': False, 'error': 'No face detected'}), 400

    cv2.imwrite(raw_path, gray)

    # Latih & evaluasi ulang seluruh dataset
    metrics = train_and_evaluate()

    return jsonify({
        'success': True,
        'message': f'Face registered for {user_id}',
        'metrics': metrics
    })


@app.route("/verify_face", methods=["POST"])
def verify_face():
    """
    Endpoint untuk verifikasi muka:
    - Simpan sementara + preprocess
    - Muat model & label_map
    - Lakukan predict dan kembalikan user_id + confidence
    """
    image = request.files.get("image")
    if not image:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    tmp_path = "tmp_verify.jpg"
    image.save(tmp_path)
    gray = detect_and_crop(tmp_path)
    os.remove(tmp_path)

    if gray is None:
        return jsonify({'success': False, 'error': 'No face detected'}), 400

    model, label_map = load_model_and_labels()
    if model is None or not label_map:
        return jsonify({'success': False, 'error': 'Model not trained yet'}), 400

    label, confidence = model.predict(gray)
    user_id = label_map.get(label, "Unknown")

    return jsonify({
        'success': True,
        'user_id': user_id,
        'confidence': float(confidence)
    })


# ─── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Jangan jalankan debug=True di production
    app.run(host="0.0.0.0", port=8000, debug=False)
