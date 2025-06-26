from flask import Flask, request, jsonify
import os
import cv2
import numpy as np

app = Flask(__name__)

# Folder untuk simpan foto wajah (training set)
FACES_DIR = "faces"
MODEL_PATH = "lbph_model.xml"

# Pastikan folder exist
os.makedirs(FACES_DIR, exist_ok=True)

# Helper: simpan foto wajah
def save_face_image(user_id, image_file):
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    # Simpan file, pakai nama urut agar bisa multi
    count = len(os.listdir(user_dir))
    filename = os.path.join(user_dir, f"{count+1}.jpg")
    image_file.save(filename)
    return filename

# Helper: load data & train model
def train_lbph_model():
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    for user_id in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, user_id)
        if not os.path.isdir(user_dir):
            continue
        if user_id not in label_map:
            label_map[user_id] = current_label
            current_label += 1
        label = label_map[user_id]
        for file in os.listdir(user_dir):
            img_path = os.path.join(user_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(label)
    if not faces:
        return None, None
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))
    model.write(MODEL_PATH)
    # Simpan mapping label-user_id
    with open("labels_map.txt", "w") as f:
        for user, lbl in label_map.items():
            f.write(f"{lbl}:{user}\n")
    return model, label_map

# Helper: load model & label map
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(MODEL_PATH)
    label_map = {}
    if os.path.exists("labels_map.txt"):
        with open("labels_map.txt") as f:
            for line in f:
                lbl, user = line.strip().split(":")
                label_map[int(lbl)] = user
    return model, label_map

# Endpoint: Register Face (training)
@app.route('/register_face', methods=['POST'])
def register_face():
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'Missing user_id or image'}), 400
    # Convert to grayscale if needed
    file_path = save_face_image(user_id, image)
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file_path, gray)
    # Retrain model with new data
    model, _ = train_lbph_model()
    if model is None:
        return jsonify({'success': False, 'error': 'Training failed'})
    return jsonify({'success': True, 'message': f'Face registered for {user_id}'})

# Endpoint: Verify Face (pencocokan)
@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    # Simpan file sementara
    tmp_path = "tmp_verify.jpg"
    image.save(tmp_path)
    img = cv2.imread(tmp_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load model & labels
    model, label_map = load_model_and_labels()
    if model is None or not label_map:
        return jsonify({'success': False, 'error': 'Model not trained yet'})
    # Prediksi
    label, confidence = model.predict(gray)
    user_id = label_map.get(label, "Unknown")
    os.remove(tmp_path)
    # Return hasil
    return jsonify({'success': True, 'user_id': user_id, 'confidence': float(confidence)})

# Endpoint root untuk test
@app.route('/')
def home():
    return "BSD Media LBPH Backend siap!"

if __name__ == "__main__":
    app.run(debug=True)
