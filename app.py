# app.py (Versi Final untuk Debugging Startup Error)

import os
import traceback
import json

# --- Blok TRY utama untuk menangkap semua error saat startup ---
try:
    # Semua import diletakkan di dalam try block untuk menangkap error jika ada
    import requests
    import cv2
    from flask import Flask, request, jsonify, send_from_directory

    from face_preprocessing import detect_and_crop
    from face_data import train_and_evaluate
    from config import FACES_DIR, MODEL_PATH, LABEL_MAP

    import firebase_admin
    from firebase_admin import credentials, storage, firestore

    print("[DEBUG] Semua library berhasil diimpor.")

    # --- Konstanta ---
    FIREBASE_BUCKET_NAME = "db-ta-bsd-media.firebasestorage.app"
    DRIVE_API_KEY = "AIzaSyC_vPd6yPwYQ60Pn-tuR3Nly_7mgXZcxGk"

    # --- Load credential dari environment variable ---
    print("[DEBUG] Membaca kredensial Firebase dari environment variable...")
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    print("[DEBUG] Kredensial Firebase berhasil dibaca.")

    # --- Inisialisasi Firebase ---
    print("[DEBUG] Inisialisasi Firebase Admin SDK...")
    # Cek apakah aplikasi sudah diinisialisasi sebelumnya untuk menghindari error
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET_NAME})
    
    bucket = storage.bucket()
    db = firestore.client()
    print("[DEBUG] Firebase Admin SDK berhasil diinisialisasi.")

    # --- Inisialisasi Aplikasi Flask ---
    app = Flask(__name__)

    # --- Error Handler Global untuk Route ---
    @app.errorhandler(Exception)
    def handle_exceptions(e):
        tb = traceback.format_exc()
        print("===== Exception Traceback dalam Route =====")
        print(tb)
        return jsonify({'success': False, 'error': str(e)}), 500

    # --- Konstanta Lokal dan Direktori ---
    os.makedirs(FACES_DIR, exist_ok=True)

    # --- Helper Functions (Lengkap) ---
    def upload_to_firebase(local_file, user_id, filename):
        blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
        blob.upload_from_filename(local_file)
        blob.make_public()
        return blob.public_url

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

    # --- Routes (Lengkap) ---
    @app.route("/", methods=["GET"])
    def home():
        return "BSD Media LBPH Backend siap!"

    @app.route('/register_face', methods=['POST'])
    def register_face():
        user_id = request.form.get('user_id')
        image   = request.files.get('image')
        if not user_id or not image: return jsonify({'success': False, 'error': 'user_id atau image tidak ada'}), 400
        raw_path = save_face_image(user_id, image)
        cropped  = detect_and_crop(raw_path)
        if cropped is None or cropped.size == 0: return jsonify({'success': False, 'error': 'Gagal cropping'}), 400
        cv2.imwrite(raw_path, cropped)
        firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))
        metrics = train_and_evaluate()
        return jsonify({ 'success': True, 'metrics': metrics, 'firebase_image_url': firebase_url })

    @app.route('/verify_face', methods=['POST'])
    def verify_face():
        image = request.files.get('image')
        if not image: return jsonify({'success': False, 'error': 'image tidak ada'}), 400
        tmp = 'tmp.jpg'; image.save(tmp)
        gray = detect_and_crop(tmp); os.remove(tmp)
        if gray is None: return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi'}), 400
        model, lblmap = load_model_and_labels()
        if model is None: return jsonify({'success': False, 'error': 'Model belum ada'}), 400
        label, conf = model.predict(gray)
        return jsonify({ 'success': True, 'user_id': lblmap.get(label, 'unknown'), 'confidence': float(conf) })

    @app.route('/find_my_photos', methods=['POST'])
    def find_my_photos():
        # Versi dummy untuk tes startup error
        print("===== MULAI find_my_photos (VERSI DUMMY TEST) =====")
        image = request.files.get('image')
        if not image:
            print("[DUMMY] Request tidak ada gambar.")
            return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
        print("[DUMMY] Gambar diterima. Langsung mengembalikan hasil palsu.")
        dummy_urls = ["https://i.imgur.com/g3OhR3s.jpeg"]
        return jsonify({'success': True, 'user_id': 'user_dummy_test', 'photo_urls': dummy_urls})

    @app.route('/list_user_faces', methods=['GET'])
    def list_user_faces():
        user_id = request.args.get('user_id')
        if not user_id: return jsonify({'success': False, 'error': 'Parameter user_id wajib diisi'}), 400
        user_dir = os.path.join(FACES_DIR, user_id)
        if not os.path.exists(user_dir): return jsonify({'success': False, 'files': [], 'error': 'User belum punya data'}), 200
        files = [f for f in os.listdir(user_dir) if f.lower().endswith('.jpg')]
        return jsonify({'success': True, 'files': files}), 200

    @app.route('/get_face_image', methods=['GET'])
    def get_face_image():
        user_id = request.args.get('user_id')
        filename = request.args.get('filename')
        if not user_id or not filename: return jsonify({'success': False, 'error': 'Parameter user_id dan filename wajib'}), 400
        user_dir = os.path.join(FACES_DIR, user_id)
        file_path = os.path.join(user_dir, filename)
        if not os.path.exists(file_path): return jsonify({'success': False, 'error': 'File tidak ditemukan'}), 404
        return send_from_directory(user_dir, filename)
    
    print("[DEBUG] Semua fungsi dan route berhasil didefinisikan.")

    # --- Menjalankan Server ---
    if __name__ == '__main__':
        print("[DEBUG] Memulai server Flask...")
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port)

# --- Blok EXCEPT untuk menangkap error saat startup ---
except Exception as e:
    # Cetak error fatal ke log utama yang bisa kita lihat
    print("\n\n===== [FATAL STARTUP ERROR] =====")
    print("Server gagal dijalankan karena error berikut:")
    traceback.print_exc()
    print("=================================\n\n")