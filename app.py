import os
import io
import json
import traceback
import numpy as np
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory
import time
from werkzeug.utils import secure_filename
from PIL import Image

from face_preprocessing import detect_and_crop
from face_data import update_lbph_model_incrementally, train_and_evaluate_full_dataset, load_model_and_labels
from config import FACES_DIR, MODEL_PATH, LABEL_MAP
from gdrive_match import find_matching_photos, find_all_matching_photos, get_all_gdrive_folder_ids

# === Firebase Admin SDK Initialization ===
import firebase_admin
from firebase_admin import credentials, storage, firestore

if not firebase_admin._apps:
    try:
        cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        cred = credentials.Certificate(cred_info)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'db-ta-bsd-media.appspot.com'
        })
        print("Firebase Admin SDK initialized successfully.")
    except KeyError:
        print("ERROR: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON not set.")
        print("Firebase Admin SDK will not be initialized. Check your Railway environment variables.")
    except Exception as e:
        print(f"ERROR initializing Firebase Admin SDK: {e}")
        traceback.print_exc()

db = firestore.client()
bucket = storage.bucket()

app = Flask(__name__)

# ==== Konstanta ====
os.makedirs(FACES_DIR, exist_ok=True)
TRAINED_MODELS_DIR = 'trained_models'
LBPH_CONFIDENCE_THRESHOLD = 60  # Bisa diubah sesuai kebutuhan

# ==== Error Handler ====
@app.errorhandler(Exception)
def handle_exceptions(e):
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

# ==== Helper ====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def upload_to_firebase(local_file, user_id, filename):
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
    return blob.public_url

def download_file_from_url(url, destination):
    try:
        print(f"DEBUG: Mengunduh {url} ke {destination}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"DEBUG: Berhasil mengunduh {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Gagal mengunduh file dari URL {url}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Terjadi error saat menyimpan file {destination}: {e}")
        return False

@app.before_first_request
def initial_model_check():
    print("INFO: Memeriksa keberadaan model LBPH saat startup...")
    MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flbph_model.xml?alt=media&token=26656ed8-3cd1-4220-a07d-aad9aaeb91f5"
    LABEL_MAP_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flabels_map.txt?alt=media&token=2ab5957f-78b2-41b0-a1aa-b2f1b8675f54"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP):
        print("INFO: Model LBPH atau label map tidak ditemukan secara lokal. Mencoba mengunduh.")
        model_downloaded = download_file_from_url(MODEL_URL, MODEL_PATH)
        labels_downloaded = download_file_from_url(LABEL_MAP_URL, LABEL_MAP)
        if not model_downloaded or not labels_downloaded:
            print("CRITICAL ERROR: Gagal mengunduh model atau label map. Aplikasi mungkin tidak berfungsi dengan baik.")
        else:
            print("INFO: Model LBPH dan label map berhasil diunduh.")
    else:
        print("INFO: Model LBPH dan label map ditemukan secara lokal.")

def save_face_image(user_id: str, image_file) -> str:
    user_dir = os.path.join(FACES_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = str(int(time.time() * 1000))
    dst = os.path.join(user_dir, f"{timestamp}.jpg")
    print(f"DEBUG: Menerima gambar untuk user '{user_id}', akan disimpan ke: {dst}")
    try:
        image_file.save(dst)
        print(f"DEBUG: Gambar berhasil disimpan ke {dst}")
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            print(f"ERROR: File {dst} kosong atau tidak ditemukan setelah disimpan.")
            return None
        return dst
    except Exception as e:
        print(f"ERROR: Gagal menyimpan gambar ke {dst}: {e}")
        traceback.print_exc()
        return None

# ==== ROUTES ====
@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

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

@app.route('/register_face', methods=['POST'])
def register_face():
    print("===== MULAI register_face =====")
    user_id = request.form.get('user_id')
    image   = request.files.get('image')
    if not user_id or not image:
        print("ERROR: user_id atau image tidak ada di request.")
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada di request'}), 400
    raw_path = save_face_image(user_id, image)
    if raw_path is None:
        return jsonify({'success': False, 'error': 'Gagal menyimpan gambar yang diunggah'}), 500
    try:
        debug_firebase_url = upload_to_firebase(raw_path, user_id, f"debug_raw_{os.path.basename(raw_path)}")
        print(f"DEBUG: Gambar mentah diupload ke Firebase untuk debug: {debug_firebase_url}")
    except Exception as e:
        print(f"ERROR: Gagal mengupload gambar mentah ke Firebase: {e}")
    try:
        firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))
        print(f"DEBUG: Gambar diupload ke Firebase: {firebase_url}")
    except Exception as e:
        print(f"ERROR: Gagal mengupload gambar ke Firebase: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Gagal mengupload gambar ke Firebase'}), 500
    print("DEBUG: Memulai update model LBPH secara incremental...")
    success = update_lbph_model_incrementally(raw_path, user_id)
    try:
        if os.path.exists(raw_path):
            os.remove(raw_path)
            print(f"DEBUG: Menghapus file sementara: {raw_path}")
    except Exception as e:
        print(f"WARNING: Gagal menghapus file sementara {raw_path}: {e}")
    if not success:
        return jsonify({'success': False, 'error': 'Gagal mengupdate model LBPH'}), 500
    print("INFO: Register face & update model berhasil.")
    return jsonify({ 'success': True, 'firebase_image_url': firebase_url })

@app.route('/verify_face', methods=['POST'])
def verify_face():
    image = request.files.get('image')
    if not image:
        return jsonify({'success': False, 'error': 'image tidak ada di request'}), 400
    tmp_filename = 'tmp_verify.jpg'
    os.makedirs(os.path.dirname(tmp_filename) or '.', exist_ok=True)
    try:
        image.save(tmp_filename)
        print(f"DEBUG: Gambar verifikasi disimpan sementara ke: {tmp_filename}")
        if not os.path.exists(tmp_filename) or os.path.getsize(tmp_filename) == 0:
            print(f"ERROR: File sementara {tmp_filename} kosong atau tidak ditemukan.")
            return jsonify({'success': False, 'error': 'Gagal memuat gambar verifikasi: file kosong'}), 400
        gray = detect_and_crop(tmp_filename)
        if gray is None:
            print("WARNING: Wajah tidak terdeteksi atau gambar kosong saat verifikasi.")
            return jsonify({'success': False, 'error': 'Wajah tidak terdeteksi atau gambar kosong'}), 400
        model, lblmap = load_model_and_labels()
        if model is None or not lblmap:
            print("ERROR: Model LBPH belum ada atau label map kosong. Tidak dapat melakukan verifikasi.")
            return jsonify({'success': False, 'error': 'Model belum ada atau label map kosong. Harap registrasi wajah terlebih dahulu.'}), 400
        if not lblmap:
            print("WARNING: Label map kosong setelah dimuat. Tidak ada wajah terdaftar untuk verifikasi.")
            return jsonify({'success': False, 'error': 'Model kosong, tidak ada wajah terdaftar untuk verifikasi.'}), 400
        label, conf = model.predict(gray)
        print(f"DEBUG: Predicted label: {label}, Confidence: {conf}")
        if label == -1 or conf > 90:
            print(f"INFO: Wajah tidak dikenali atau confidence terlalu rendah (Label: {label}, Conf: {conf}).")
            return jsonify({'success': False, 'error': 'Wajah tidak dikenali atau confidence terlalu rendah.'}), 404
        recognized_user_id = lblmap.get(label)
        if recognized_user_id is None:
            print(f"ERROR: Label terprediksi ({label}) tidak ditemukan di label map.")
            return jsonify({'success': False, 'error': 'Label terprediksi tidak ditemukan di label map.'}), 500
        print(f"INFO: Verifikasi berhasil. User ID: {recognized_user_id}, Confidence: {conf}")
        return jsonify({ 'success': True, 'user_id': recognized_user_id, 'confidence': float(conf) })
    except Exception as e:
        print(f"ERROR: Terjadi error saat verify_face: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Terjadi error internal: {e}'}), 500
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
            print(f"DEBUG: Menghapus file sementara: {tmp_filename}")

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
        print("\n===== Endpoint /find_my_photos DIPANGGIL (MULTI-MODE - VERSI TERBARU) =====\n")
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'File gambar tidak ditemukan'}), 400

        image = request.files['image']
        user_tmp = 'tmp_user_search.jpg'
        os.makedirs(os.path.dirname(user_tmp) or '.', exist_ok=True)
        image.save(user_tmp)
        print(f"DEBUG: Gambar klien untuk pencarian disimpan sementara ke: {user_tmp}")

        import cv2
        import numpy as np
        img = cv2.imread(user_tmp)
        if img is None:
            print("ERROR: Gagal memuat file gambar klien untuk pencarian.")
            return jsonify({'success': False, 'error': 'Gagal memuat file gambar klien'}), 400

        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)
            print("DEBUG: Gambar klien dikonversi ke format 8-bit.")
        cv2.imwrite(user_tmp, img)
        print(f"DEBUG: Gambar klien yang sudah bersih disimpan kembali ke: {user_tmp}")

        # --- MODE 1: Filter folder tertentu (search.dart) ---
        drive_links = request.form.get('drive_links')
        if drive_links:
            try:
                drive_folders = json.loads(drive_links)
                if not isinstance(drive_folders, list):
                    raise ValueError("drive_links harus list string")
            except Exception as e:
                return jsonify({'success': False, 'error': 'drive_links harus list string'}), 400

            print(f"DEBUG: Filter pencarian hanya di folder Google Drive berikut: {drive_folders}")
            matched_photos = []
            folder_data = get_all_gdrive_folder_ids()
            folderid_to_docid = {fd['drive_folder_id']: fd['firestore_doc_id'] for fd in folder_data}
            for link in drive_folders:
                if 'folders/' in link:
                    folder_id = link.split('folders/')[1].split('?')[0]
                    session_id = folderid_to_docid.get(folder_id)
                    matches = find_matching_photos(user_tmp, folder_id, session_id)
                    for m in matches:
                        m['sessionId'] = session_id
                    matched_photos.extend(matches)
            print("INFO: Pencarian foto berdasarkan drive_links selesai. Mengirimkan respons.")
            return jsonify({'success': True, 'matched_photos': matched_photos})

        # --- MODE 2: Cari seluruh database fotografer (home.dart) ---
        print("DEBUG: Tidak ada drive_links, mencari ke seluruh database sesi.")
        all_folder_data = get_all_gdrive_folder_ids()
        print(f"DEBUG: Ditemukan {len(all_folder_data)} folder Google Drive.")
        matches = find_all_matching_photos(user_tmp, all_folder_data)
        for m in matches:
            session_id = m.get('sessionId') or m.get('folder_id')
            m['sessionId'] = session_id
        print("INFO: Pencarian foto seluruh database selesai. Mengirimkan respons.")
        return jsonify({'success': True, 'matched_photos': matches})

    except Exception as e:
        print("===== ERROR TRACEBACK in /find_my_photos =====")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if os.path.exists('tmp_user_search.jpg'):
            os.remove('tmp_user_search.jpg')
            print(f"DEBUG: Menghapus file sementara: tmp_user_search.jpg")

@app.route('/debug_ls', methods=['GET'])
def debug_ls():
    result = {}
    try:
        result['root'] = os.listdir('.')
    except Exception as e:
        result['root'] = str(e)
    try:
        result['faces_dir'] = os.listdir(FACES_DIR)
    except Exception as e:
        result['faces_dir'] = str(e)
    user_id_param = request.args.get('user_id')
    if user_id_param:
        user_specific_dir = os.path.join(FACES_DIR, user_id_param)
        try:
            result[f'faces/{user_id_param}'] = os.listdir(user_specific_dir)
        except Exception as e:
            result[f'faces/{user_id_param}'] = str(e)
    result['model_path_exists'] = os.path.exists(MODEL_PATH)
    result['label_map_path_exists'] = os.path.exists(LABEL_MAP)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))