import os
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

def upload_to_firebase(local_path, user_id, filename):
    # Contoh implementasi, sesuaikan dengan SDK cloud storage kamu (misal: Google Cloud Storage)
    from google.cloud import storage
    bucket_name = "db-ta-bsd-media.firebasestorage.app"  # Ganti dengan nama bucket kamu
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f'face-dataset/{user_id}/{filename}')
    blob.upload_from_filename(local_path)

def update_lbph_model_incrementally(image_path, user_id):
    # 1. Load gambar dan ekstrak wajah (ROI) dengan MTCNN
    pil_img = Image.open(image_path).convert('RGB')
    mtcnn_detector = MTCNN()
    faces = mtcnn_detector.detect_faces(np.array(pil_img))
    if not faces:
        print("ERROR: Tidak ada wajah terdeteksi pada gambar (MTCNN).")
        return False

    x, y, w, h = faces[0]['box']
    # Perbaiki jika bounding box negatif
    x = max(0, x)
    y = max(0, y)
    face_roi = pil_img.crop((x, y, x + w, y + h)).resize((100, 100)).convert('L')
    face_roi = np.array(face_roi)

    # 2. Siapkan path model & label map
    os.makedirs("trained_models", exist_ok=True)
    model_path = f"trained_models/{user_id}_lbph.yml"
    label_map_path = f"trained_models/{user_id}_labels.npy"

    # 3. Load model & label map jika sudah ada, jika tidak buat baru
    if os.path.exists(model_path) and os.path.exists(label_map_path):
        lbph_model = cv2.face.LBPHFaceRecognizer_create()
        lbph_model.read(model_path)
        label_map = np.load(label_map_path, allow_pickle=True).item()
        # Tambah data baru
        if user_id not in label_map.values():
            next_label = max(label_map.keys()) + 1
            label_map[next_label] = user_id
        label_ids = [k for k, v in label_map.items() if v == user_id]
        label_id = label_ids[0] if label_ids else max(label_map.keys()) + 1
        lbph_model.update([face_roi], np.array([label_id]))
    else:
        lbph_model = cv2.face.LBPHFaceRecognizer_create()
        label_map = {0: user_id}
        lbph_model.train([face_roi], np.array([0]))

    # 4. Simpan model & label map ke lokal
    lbph_model.save(model_path)
    np.save(label_map_path, label_map)

    # 5. Upload model & label map ke cloud storage (face-dataset/{user_id}/)
    try:
        upload_to_firebase(model_path, user_id, f"{user_id}_lbph.yml")
        upload_to_firebase(label_map_path, user_id, f"{user_id}_labels.npy")
        print("INFO: Model & label map berhasil diupload ke cloud storage.")
    except Exception as e:
        print(f"WARNING: Gagal upload model/label map ke cloud storage: {e}")

    return True