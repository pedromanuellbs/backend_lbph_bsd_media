# worker.py
import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
import requests

from gdrive_match import find_all_matching_photos, get_all_gdrive_folder_ids

print("--- Worker Mulai Berjalan ---")

# --- Inisialisasi Firebase ---
if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

db = firestore.client()
jobs_collection = db.collection('photo_search_jobs')

def process_job(job_doc):
    job_id = job_doc.id
    job_data = job_doc.to_dict()
    job_ref = jobs_collection.document(job_id)

    print(f"Mengambil tugas: {job_id}")
    
    # Tandai sebagai 'processing' agar tidak diambil worker lain
    job_ref.update({'status': 'processing'})

    try:
        # Download gambar klien dari URL di Firebase Storage
        client_image_url = job_data['clientImageURL']
        response = requests.get(client_image_url, stream=True)
        response.raise_for_status() # Cek jika download error
        
        tmp_user_path = f"/tmp/{job_id}_client.jpg"
        with open(tmp_user_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Gambar klien untuk job {job_id} berhasil diunduh ke {tmp_user_path}")

        # Jalankan logika pencarian foto yang sudah ada
        all_folder_ids = get_all_gdrive_folder_ids()
        matches = find_all_matching_photos(tmp_user_path, all_folder_ids, threshold=0.4)
        
        print(f"Pencarian selesai untuk job {job_id}. Ditemukan {len(matches)} foto.")

        # Update job dengan hasil dan status 'completed'
        job_ref.update({
            'status': 'completed',
            'results': matches,
            'completedAt': firestore.SERVER_TIMESTAMP
        })
        
        os.remove(tmp_user_path)

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada job {job_id}: {e}")
        # Update job dengan status 'failed' dan pesan error
        job_ref.update({
            'status': 'failed',
            'error': str(e),
            'completedAt': firestore.SERVER_TIMESTAMP
        })

# --- Loop Utama Worker ---
while True:
    try:
        # Cari satu tugas yang statusnya 'pending'
        pending_jobs = jobs_collection.where('status', '==', 'pending').limit(1).stream()
        
        job_found = False
        for job in pending_jobs:
            job_found = True
            process_job(job)
            break # Proses satu per satu
        
        if not job_found:
            # Jika tidak ada tugas, tunggu sebelum cek lagi
            print("Tidak ada tugas baru, menunggu 10 detik...")
            time.sleep(10)

    except Exception as e:
        print(f"[FATAL] Worker loop error: {e}")
        time.sleep(30) # Tunggu lebih lama jika ada error fatal