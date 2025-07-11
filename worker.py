# worker.py (VERSI SIMPLE UNTUK SKRIPSI + PROGRESS BAR)

import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import cv2
import numpy as np

# --- ASUMSI: Anda punya fungsi-fungsi ini dari project Anda ---
# Fungsi ini harus mengambil daftar semua foto dari Google Drive Anda
from gdrive_match import get_all_photos_from_gdrive 
# Fungsi ini membandingkan 2 wajah dan mengembalikan True jika cocok
from gdrive_match import compare_faces 
from face_preprocessing import detect_and_crop

print("--- Worker (Versi Skripsi) Dimulai ---")

# --- Inisialisasi Firebase ---
if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

db = firestore.client()
jobs_collection = db.collection('photo_search_jobs')

def download_image(url, save_path):
    """Helper untuk download gambar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def process_job_with_progress():
    """Mencari satu tugas, memprosesnya dengan laporan progres."""
    
    job_ref = None # Definisikan di luar try-block
    
    # Cari satu tugas yang statusnya 'pending'
    pending_jobs_query = jobs_collection.where('status', '==', 'pending').limit(1)
    
    try:
        docs = list(pending_jobs_query.stream())
        if not docs:
            print("Tidak ada tugas baru.")
            return

        job_doc = docs[0]
        job_id = job_doc.id
        job_data = job_doc.to_dict()
        job_ref = jobs_collection.document(job_id)

        print(f"Mengambil tugas: {job_id}")
        
        # 1. Download dan proses wajah klien (hanya sekali)
        client_image_url = job_data['clientImageURL']
        tmp_client_path = f"/tmp/{job_id}_client.jpg"
        download_image(client_image_url, tmp_client_path)
        client_face = detect_and_crop(tmp_client_path)
        
        if client_face is None:
            raise ValueError("Wajah tidak terdeteksi pada gambar klien.")

        # 2. Dapatkan daftar semua foto dari Google Drive
        all_photos = get_all_photos_from_gdrive()
        total_photos = len(all_photos)
        
        # Update status dengan total foto
        job_ref.update({
            'status': 'processing',
            'progress': 0,
            'total': total_photos
        })
        
        matches = []
        
        # 3. Mulai loop pencarian satu per satu
        for i, photo_data in enumerate(all_photos):
            photo_id = photo_data['id']
            photo_url = photo_data['url']
            
            print(f"  Memproses {i + 1}/{total_photos} : {photo_id}")
            
            try:
                # Download foto dari drive
                tmp_gdrive_path = f"/tmp/{photo_id}.jpg"
                download_image(photo_url, tmp_gdrive_path)
                
                # Deteksi wajah di foto dari drive
                gdrive_face = detect_and_crop(tmp_gdrive_path)
                
                if gdrive_face is not None:
                    # Bandingkan wajah
                    # Anda perlu `compare_faces` yang menggunakan model.predict
                    is_match = compare_faces(client_face, gdrive_face) 
                    if is_match:
                        print(f"    -> DITEMUKAN KECOCOKAN: {photo_id}")
                        matches.append({"url": photo_url, "id": photo_id})
                
                os.remove(tmp_gdrive_path)

            except Exception as photo_error:
                print(f"    -> Gagal memproses foto {photo_id}: {photo_error}")
                continue

            # 4. KIRIM UPDATE PROGRESS SETELAH SETIAP FOTO!
            job_ref.update({'progress': i + 1})

        # 5. Selesai, update hasil akhir
        print(f"Pencarian selesai untuk job {job_id}. Ditemukan {len(matches)} foto.")
        job_ref.update({
            'status': 'completed',
            'results': matches,
            'completedAt': firestore.SERVER_TIMESTAMP
        })
        os.remove(tmp_client_path)

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada job: {e}")
        if job_ref:
            job_ref.update({'status': 'failed', 'error': str(e)})

# --- Jalankan proses ---
if __name__ == "__main__":
    process_job_with_progress()