# worker.py (VERSI FINAL)

import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
import requests

from gdrive_match import find_all_matching_photos, get_all_gdrive_folder_ids

print("--- Worker Dimulai ---")

# --- Inisialisasi Firebase ---
if not firebase_admin._apps:
    # ... (kode inisialisasi firebase Anda tetap sama) ...
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'db-ta-bsd-media.firebasestorage.app'
    })

db = firestore.client()
jobs_collection = db.collection('photo_search_jobs')

def process_single_job():
    """Mencari satu tugas 'pending', memprosesnya, lalu selesai."""
    
    # Cari satu tugas yang statusnya 'pending'
    pending_jobs_query = jobs_collection.where('status', '==', 'pending').limit(1)
    
    try:
        docs = list(pending_jobs_query.stream())
        if not docs:
            print("Tidak ada tugas baru. Worker akan berhenti.")
            return

        job_doc = docs[0]
        job_id = job_doc.id
        job_data = job_doc.to_dict()
        job_ref = jobs_collection.document(job_id)

        print(f"Mengambil tugas: {job_id}")
        job_ref.update({'status': 'processing'})

        # Download gambar klien
        client_image_url = job_data['clientImageURL']
        response = requests.get(client_image_url, stream=True)
        response.raise_for_status()
        
        tmp_user_path = f"/tmp/{job_id}_client.jpg"
        with open(tmp_user_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Jalankan logika pencarian foto
        all_folder_ids = get_all_gdrive_folder_ids()
        matches = find_all_matching_photos(tmp_user_path, all_folder_ids, threshold=0.593)
        
        print(f"Pencarian selesai untuk job {job_id}. Ditemukan {len(matches)} foto.")

        # Update job dengan hasil
        job_ref.update({
            'status': 'completed',
            'results': matches,
            'completedAt': firestore.SERVER_TIMESTAMP
        })
        os.remove(tmp_user_path)

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada job {job_id}: {e}")
        if 'job_ref' in locals():
            job_ref.update({'status': 'failed', 'error': str(e)})

# --- Jalankan proses HANYA SATU KALI ---
if __name__ == "__main__":
    process_single_job()
    print("--- Tugas Selesai. Worker berhenti. ---")