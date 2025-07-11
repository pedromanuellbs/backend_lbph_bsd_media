# worker.py (Diperbarui untuk LBPH)

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
from face_preprocessing import detect_and_crop # Pastikan ini diimpor dengan benar

print("--- Worker (Versi Skripsi Final LBPH) Dimulai ---")

# --- Inisialisasi Firebase ---
if not firebase_admin._apps:
    cred_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    cred = credentials.Certificate(cred_info)
    firebase_admin.initialize_app(cred)

db = firestore.client()
jobs_collection = db.collection('photo_search_jobs')

def download_image_from_url(url):
    """
    Helper untuk download gambar dari URL publik dan mengembalikannya sebagai data gambar (NumPy array).
    Digunakan untuk mengunduh gambar klien dari Firebase Storage.
    """
    response = requests.get(url)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # Membaca sebagai gambar berwarna
    return image

def process_single_job():
    """Mencari satu tugas, memprosesnya dengan laporan progres, lalu berhenti."""
    job_ref = None
    # Mengambil satu tugas yang statusnya 'pending'
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
        
        # 1. Unduh dan proses wajah klien (hanya sekali)
        client_image_url = job_data['clientImageURL']
        
        # Unduh gambar klien sebagai NumPy array
        client_raw_image = download_image_from_url(client_image_url)
        
        # Deteksi dan crop wajah klien. Ini akan mengembalikan NumPy array grayscale.
        client_face_cropped = detect_and_crop(client_raw_image)
        
        if client_face_cropped is None:
            raise ValueError("Wajah tidak terdeteksi pada gambar klien.")

        # 2. Dapatkan daftar semua foto dari Google Drive
        all_photos = get_all_photo_files()
        total_photos = len(all_photos)
        
        # Perbarui status dengan total foto
        job_ref.update({
            'status': 'processing', 
            'progress': 0, 
            'total': total_photos
        })
        
        matches = []
        drive_service = get_drive_service() # Inisialisasi layanan Drive sekali
        
        # 3. Mulai loop pencarian satu per satu
        for i, photo_data in enumerate(all_photos):
            photo_id = photo_data['id']
            photo_name = photo_data['name'] # Ambil nama file untuk logging
            print(f"  Memproses {i + 1}/{total_photos} : {photo_name}")
            
            try:
                # Unduh foto dari Drive sebagai NumPy array
                gdrive_raw_image = download_drive_photo(drive_service, photo_id)
                if gdrive_raw_image is None:
                    print(f"    - Gagal mengunduh gambar dari Drive: {photo_name}")
                    continue
                
                # Deteksi dan crop wajah di foto dari Drive. Ini akan mengembalikan NumPy array grayscale.
                gdrive_face_cropped = detect_and_crop(gdrive_raw_image)
                
                if gdrive_face_cropped is not None:
                    # Bandingkan wajah menggunakan fungsi compare_faces yang sudah diubah
                    # Meneruskan NumPy array langsung ke compare_faces
                    is_match = compare_faces(client_face_cropped, gdrive_face_cropped) 
                    if is_match:
                        print(f"    -> DITEMUKAN KECOCOKAN: {photo_name} (ID: {photo_id})")
                        public_url = f'https://drive.google.com/uc?export=view&id={photo_id}'
                        matches.append({"url": public_url, "id": photo_id, "name": photo_name})
                else:
                    print(f"    - Tidak ada wajah terdeteksi di foto Google Drive: {photo_name}")
                
            except Exception as photo_error:
                print(f"    - Gagal memproses foto {photo_name} (ID: {photo_id}): {photo_error}")
                # import traceback; traceback.print_exc() # Aktifkan untuk debugging
                continue

            # 4. KIRIM UPDATE PROGRESS SETELAH SETIAP FOTO!
            job_ref.update({'progress': i + 1})

        # 5. Selesai, perbarui hasil akhir
        print(f"--- Pencarian selesai untuk job {job_id}. Ditemukan {len(matches)} foto. ---")
        job_ref.update({
            'status': 'completed',
            'results': matches,
            'completedAt': firestore.SERVER_TIMESTAMP
        })

    except Exception as e:
        print(f"[FATAL ERROR] Terjadi kesalahan pada job {job_id}: {e}")
        # import traceback; traceback.print_exc() # Aktifkan untuk debugging
        if job_ref:
            job_ref.update({'status': 'failed', 'error': str(e)})

# --- Jalankan proses secara berkelanjutan (DIREKOMENDASIKAN UNTUK PRODUKSI) ---
if __name__ == "__main__":
    while True:
        try:
            process_single_job()
        except Exception as e:
            print(f"[LOOP ERROR] Terjadi kesalahan saat memproses tugas, mencoba lagi: {e}")
        
        print("Menunggu tugas baru (5 detik)...")
        time.sleep(5) # Tunggu sebentar sebelum mencari tugas lagi
