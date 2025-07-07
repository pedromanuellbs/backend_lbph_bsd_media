import os
import json
import cv2
from dotenv import load_dotenv
load_dotenv()

# --- Import komponen yang sudah ada dari proyek Anda ---
from face_preprocessing import detect_and_crop
from face_data import train_and_evaluate
# ----------------------------------------------------

# --- Inisialisasi Firebase Admin SDK ---
import firebase_admin
from firebase_admin import credentials, storage

print("Menginisialisasi Firebase...")
if not firebase_admin._apps:
    # --- KODE INISIALISASI DIPERBARUI ---
    # Mengambil nama file dari .env
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if cred_path is None:
        raise ValueError("Variabel GOOGLE_APPLICATION_CREDENTIALS tidak ditemukan di file .env")
    
    cred = credentials.Certificate(cred_path)
    # -------------------------------------
    
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })
print("Firebase berhasil diinisialisasi.")
# ----------------------------------------

# --- Konstanta Direktori Lokal ---
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)
# ---------------------------------

def process_new_faces():
    """
    Fungsi utama untuk mengunduh, memproses, dan melatih ulang wajah baru.
    """
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="face-dataset/")
    
    new_faces_processed_count = 0
    
    print("Mengecek file baru di Firebase Storage...")

    for blob in blobs:
        if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        parts = blob.name.split('/')
        if len(parts) < 3:
            continue
            
        user_id = parts[1]
        filename = parts[2]
        
        local_user_dir = os.path.join(FACES_DIR, user_id)
        local_file_path = os.path.join(local_user_dir, filename)
        
        if os.path.exists(local_file_path):
            continue

        print(f"File baru ditemukan: {blob.name}. Memproses...")
        
        os.makedirs(local_user_dir, exist_ok=True)
        
        temp_path = os.path.join(local_user_dir, f"temp_{filename}")
        blob.download_to_filename(temp_path)
        
        cropped_face = detect_and_crop(temp_path)
        
        os.remove(temp_path)
        
        if cropped_face is not None:
            cv2.imwrite(local_file_path, cropped_face)
            print(f"✅ Berhasil memproses dan menyimpan: {local_file_path}")
            new_faces_processed_count += 1
        else:
            print(f"⚠️ Gagal mendeteksi wajah pada file: {blob.name}")

    print("-" * 30)
    
    if new_faces_processed_count > 0:
        print(f"Ditemukan {new_faces_processed_count} wajah baru. Memulai ulang training model LBPH...")
        try:
            metrics = train_and_evaluate()
            print("✅ Model berhasil dilatih ulang!")
            print(f"   Metrics: {metrics}")
        except Exception as e:
            print(f"❌ Terjadi error saat training model: {e}")
    else:
        print("Tidak ada wajah baru untuk diproses. Model sudah up-to-date.")
    
    print("Proses selesai.")


if __name__ == '__main__':
    process_new_faces()