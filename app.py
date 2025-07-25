import os
import json
import traceback
import numpy as np
import requests
import cv2
from flask import Flask, request, jsonify, send_from_directory
import time

from face_preprocessing import detect_and_crop
from face_data import update_lbph_model_incrementally, train_and_evaluate_full_dataset, load_model_and_labels

from config import FACES_DIR, MODEL_PATH, LABEL_MAP

from google.cloud import firestore

import firebase_admin
from firebase_admin import credentials, storage, firestore

from gdrive_match import find_matching_photos, find_all_matching_photos, get_all_gdrive_folder_ids

if not firebase_admin._apps:
    try:
        cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
        cred = credentials.Certificate(cred_info)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
        })
        print("Firebase Admin SDK initialized successfully.")
    except KeyError:
        print("ERROR: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON not set.")
        print("Firebase Admin SDK will not be initialized. Check your Railway environment variables.")
    except Exception as e:
        print(f"ERROR initializing Firebase Admin SDK: {e}")
        traceback.print_exc()

def upload_to_firebase(local_file, user_id, filename):
    bucket = storage.bucket()
    blob = bucket.blob(f"face-dataset/{user_id}/{filename}")
    blob.upload_from_filename(local_file)
    blob.make_public()
    return blob.public_url

def upload_model_files_to_firebase():
    try:
        print("DEBUG: Mengupload model dan label map yang diperbarui ke Firebase Storage...")
        bucket = storage.bucket()
        model_blob = bucket.blob("face-recognition-models/lbph_model.xml")
        model_blob.upload_from_filename(MODEL_PATH)
        label_map_blob = bucket.blob("face-recognition-models/labels_map.txt")
        label_map_blob.upload_from_filename(LABEL_MAP)
        print("SUCCESS: Model dan label map berhasil diupload ke Firebase Storage.")
        return True
    except Exception as e:
        print(f"ERROR: Gagal mengupload model/label map ke Firebase: {e}")
        traceback.print_exc()
        return False

app = Flask(__name__)

@app.route('/face_login', methods=['POST'])
def face_login():
    import tempfile
    import os

    uid = request.form.get('uid')
    image = request.files.get('image')
    print(f"DEBUG UID: {uid}, IMAGE: {image}")
    if not uid or not image:
        return jsonify({'success': False, 'error': 'UID atau image tidak ada'}), 400

    temp_img_path = None

    try:
        # Simpan gambar sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img_file:
            image.save(temp_img_file)
            temp_img_path = temp_img_file.name

        # Deteksi wajah saja dengan MTCNN
        def detect_faces_only(img_path):
            from PIL import Image
            import numpy as np
            from mtcnn import MTCNN
            img = Image.open(img_path).convert('RGB')
            img_np = np.asarray(img)
            detector = MTCNN()
            faces = detector.detect_faces(img_np)
            print(f"DEBUG: Jumlah wajah terdeteksi (MTCNN): {len(faces)}")
            return len(faces)

        num_faces = detect_faces_only(temp_img_path)
        if num_faces == 1:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Tidak ada atau lebih dari satu wajah terdeteksi.'}), 400

    except Exception as e:
        import traceback
        print("ERROR in /face_login:", e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

    finally:
        # Pembersihan file sementara
        try:
            if temp_img_path and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        except Exception as cleanup_error:
            print(f"WARNING: Gagal menghapus file sementara {temp_img_path}: {cleanup_error}")

def download_model_and_labels_if_needed(model_path, label_map_path):
    MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flbph_model.xml?alt=media&token=26656ed8-3cd1-4220-a07d-aad9aaeb91f5"
    LABEL_MAP_URL = "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flabels_map.txt?alt=media&token=2ab5957f-78b2-41b0-a1aa-b2f1b8675f54"
    if not os.path.exists(model_path):
        download_file_from_url(MODEL_URL, model_path)
    if not os.path.exists(label_map_path):
        download_file_from_url(LABEL_MAP_URL, label_map_path)

def detect_and_crop_face(image_path):
    # Contoh: pakai OpenCV haarcascade, atau ganti dengan MTCNN jika ada
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image.")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    print(f"Faces detected: {len(faces)}")
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    cropped = image[y:y+h, x:x+w]
    cropped_resized = cv2.resize(cropped, (96, 96))
    cropped_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
    return cropped_rgb

@app.errorhandler(Exception)
def handle_exceptions(e):
    tb = traceback.format_exc()
    print("===== Exception Traceback =====")
    print(tb)
    return jsonify({'success': False, 'error': str(e)}), 500

os.makedirs(FACES_DIR, exist_ok=True)

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

@app.route("/", methods=["GET"])
def home():
    return "BSD Media LBPH Backend siap!"

@app.route('/register_face', methods=['POST'])
def register_face():
    print("===== MULAI register_face =====")
    user_id = request.form.get('user_id')
    image = request.files.get('image')
    if not user_id or not image:
        return jsonify({'success': False, 'error': 'user_id atau image tidak ada'}), 400

    raw_path = save_face_image(user_id, image)
    if raw_path is None:
        return jsonify({'success': False, 'error': 'Gagal menyimpan gambar'}), 500

    try:
        # Upload foto ke Firebase Storage di folder user_id
        firebase_url = upload_to_firebase(raw_path, user_id, os.path.basename(raw_path))
    except Exception as e:
        return jsonify({'success': False, 'error': 'Gagal mengupload gambar ke Firebase'}), 500

    # Update/train model khusus user ini
    success = update_lbph_model_incrementally(raw_path, user_id)

    try:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    except Exception as e:
        print(f"WARNING: Gagal menghapus file sementara {raw_path}: {e}")

    if not success:
        return jsonify({'success': False, 'error': 'Gagal mengupdate model LBPH'}), 500

    # Upload model dan label map ke folder UID di Firebase Storage
    def upload_user_model_files_to_firebase(uid):
        try:
            bucket = storage.bucket()
            # Simpan model di subfolder 'model' dalam folder UID
            model_blob = bucket.blob(f"face-dataset/{uid}/model/lbph_model.xml")
            model_blob.upload_from_filename(MODEL_PATH)
            label_map_blob = bucket.blob(f"face-dataset/{uid}/model/labels_map.txt")
            label_map_blob.upload_from_filename(LABEL_MAP)
            print(f"SUCCESS: Model dan label map berhasil diupload ke Firebase Storage untuk user {uid}.")
            return True
        except Exception as e:
            print(f"ERROR: Gagal mengupload model/label map ke Firebase untuk user {uid}: {e}")
            traceback.print_exc()
            return False

    if not upload_user_model_files_to_firebase(user_id):
        print("WARNING: Registrasi wajah berhasil, tapi gagal menyimpan salinan permanen ke cloud.")

    return jsonify({'success': True, 'firebase_image_url': firebase_url})

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
    app.run(host='0.0.0.0', port=8000)