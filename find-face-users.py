import os
import io
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app, storage as fb_storage
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Ambil credentials dari ENV Railway
cred_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if not cred_json:
    raise Exception('GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set')
cred_dict = json.loads(cred_json)
cred = credentials.Certificate(cred_dict)
initialize_app(cred, {'storageBucket': 'db-ta-bsd-media.appspot.com'})

db = firestore.client()
bucket = fb_storage.bucket()

TRAINED_MODELS_DIR = 'trained_models'
LBPH_CONFIDENCE_THRESHOLD = 60  # Bisa diubah sesuai kebutuhan

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/find-face-users', methods=['POST'])
def find_face_users():
    if 'image' not in request.files or 'user_id' not in request.form:
        return jsonify({'success': False, 'error': 'image dan user_id wajib'}), 400

    file = request.files['image']
    user_id = request.form['user_id']

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Ekstensi file tidak diperbolehkan'}), 400

    model_path = os.path.join(TRAINED_MODELS_DIR, f'{user_id}_lbph.yml')
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': 'Model wajah user tidak ditemukan'}), 404

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)

    label_map_path = os.path.join(TRAINED_MODELS_DIR, f'{user_id}_labels.npy')
    if not os.path.exists(label_map_path):
        return jsonify({'success': False, 'error': 'Label map user tidak ditemukan'}), 404

    label_map = np.load(label_map_path, allow_pickle=True).item()

    in_mem_file = io.BytesIO(file.read())
    pil_image = Image.open(in_mem_file).convert('L')
    img = np.array(pil_image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400

    matched_photos = []

    photos_ref = db.collection('photos').where('owner_id', '==', user_id)
    photo_docs = photos_ref.stream()

    for photo_doc in photo_docs:
        photo_data = photo_doc.to_dict()
        photo_url = photo_data.get('url')
        storage_path = photo_data.get('storage_path')
        if not photo_url or not storage_path:
            continue

        try:
            blob = bucket.blob(storage_path)
            img_bytes = blob.download_as_bytes()
            pil_photo = Image.open(io.BytesIO(img_bytes)).convert('L')
            np_photo = np.array(pil_photo)
            faces_in_photo = face_cascade.detectMultiScale(np_photo, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_in_photo:
                face_roi = np_photo[y:y+h, x:x+w]
                try:
                    label, conf = model.predict(cv2.resize(face_roi, (img.shape[1], img.shape[0])))
                    if label_map.get(label) == user_id and conf < LBPH_CONFIDENCE_THRESHOLD:
                        matched_photos.append(photo_url)
                        break
                except Exception as e:
                    print(f'Gagal predict foto: {e}')
                    continue
        except Exception as e:
            print(f'Gagal download/olah foto dari storage: {e}')
            continue

    return jsonify({'success': True, 'matched_photos': matched_photos}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))