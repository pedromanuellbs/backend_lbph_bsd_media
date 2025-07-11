# worker.py

import os
import time
import json
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import cv2
import numpy as np

# Pastikan nama fungsi ini sesuai dengan yang ada di file gdrive_match.py Anda
from gdrive_match import get_all_photo_files, download_drive_photo, compare_faces, get_drive_service
from face_preprocessing import detect_and_crop

print("--- Worker (Versi Skripsi Final) Dimulai ---")

if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()
jobs_collection = db.collection('photo_search_jobs')

def download_image_from_url(url):
    """Helper untuk download gambar dari URL publik dan mengembalikannya sebagai data gambar."""
    response = requests.get(url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def process_single_job():
    """Mencari satu tugas, memprosesnya dengan laporan progres, lalu berhenti."""
    job_ref = None
    pending_jobs_query = jobs_collection.where('status', '==', 'pending').limit(1)
    
    try:
        docs = list(pending_jobs_query.stream())
        if not docs:
            print("Tidak ada tugas baru. Worker berhenti.")
            return

        job_doc = docs[0]
        job_id = job_doc.id
        job_data = job_doc.to_dict()
        job_ref = jobs_collection.document(job_id)

        print(f"\n--- Mengambil tugas: {job_id} ---")
        
        client_image_url = job_data['clientImageURL']
        client_image_data = download_image_from_url(client_image_url)
        client_face = detect_and_crop(client_image_data)
        
        if client_face is None:
            raise ValueError("Wajah tidak terdeteksi pada gambar klien.")

        all_photos = get_all_photo_files()
        total_photos = len(all_photos)
        
        job_ref.update({
            'status': 'processing', 'progress': 0, 'total': total_photos
        })
        
        matches = []
        drive_service = get_drive_service()
        
        for i, photo_data in enumerate(all_photos):
            photo_id = photo_data['id']
            print(f"  Memproses {i + 1}/{total_photos} : {photo_data['name']}")
            
            try:
                gdrive_image_data = download_drive_photo(drive_service, photo_id)
                if gdrive_image_data is None:
                    continue
                
                gdrive_face = detect_and_crop(gdrive_image_data)
                
                if gdrive_face is not None:
                    is_match = compare_faces(client_face, gdrive_face) 
                    if is_match:
                        print(f"    -> DITEMUKAN KECOCOKAN: {photo_id}")
                        public_url = f'https://drive.google.com/uc?export=view&id={photo_id}'
                        matches.append({"url": public_url, "id": photo_id, "name": photo_data['name']})
            
            except Exception as photo_error:
                print(f"    - Gagal memproses foto {photo_id}: {photo_error}")
                continue

            job_ref.update({'progress': i + 1})

        print(f"--- Pencarian selesai. Ditemukan {len(matches)} foto. ---")
        job_ref.update({
            'status': 'completed',
            'results': matches,
            'completedAt': firestore.SERVER_TIMESTAMP
        })

    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan: {e}")
        if job_ref:
            job_ref.update({'status': 'failed', 'error': str(e)})

# --- Jalankan proses HANYA SEKALI ---
if __name__ == "__main__":
    process_single_job()