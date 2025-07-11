from flask import Flask, request, jsonify
import os
import cv2
import logging

from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate, FACES_DIR, MODEL_PATH, LABELS_MAP_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# pastikan folder dataset ada
os.makedirs(FACES_DIR, exist_ok=True)

def save_face_image(user_id, image_file):
    """Simpan gambar upload ke folder faces/<user_id>/N.jpg"""
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    count = len(os.listdir(user_dir))
    dst = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(dst)
    return dst

def load_model_and_labels():
    """Load LBPH model dan mapping label→user_id dari filesystem."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_MAP_PATH):
        return None, {}
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    label_map = {}
    with open(LABELS_MAP_PATH) as f:
        for line in f:
            lbl, uid = line.strip().split(":")
            label_map[int(lbl)] = uid
    return model, label_map

@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get('user_id')
    image   = request.files.get('image')
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'Missing user_id or image'}), 400

    # simpan upload + crop/grayscale
    raw_path = save_face_image(user_id, image)
    cropped = detect_and_crop(raw_path)
    if cropped is None:
        return jsonify({'success': False, 'error': 'No face detected'}), 400
    # overwrite file dengan hasil crop grayscale 96×96
    cv2.imwrite(raw_path, cropped)

    # latih ulang seluruh dataset, hitung metrik, simpan model & label_map
    metrics = train_and_evaluate()
    return jsonify({
        'success': True,
        'message': f'Face registered for {user_id}',
        'metrics': metrics
    })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    # simpan sementara & preprocess
    tmp_path = "tmp_verify.jpg"
    image.save(tmp_path)
    gray = detect_and_crop(tmp_path)
    os.remove(tmp_path)
    if gray is None:
        return jsonify({'success': False, 'error': 'No face detected'}), 400

    # predict
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

@app.route('/')
def home():
    return "BSD Media LBPH Backend siap!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
