# clean_dataset.py

import os
import sys
from config import FACES_DIR
from face_preprocessing import detect_and_crop

def clean_dataset():
    """
    Melakukan iterasi ke semua gambar di folder dataset,
    dan menghapus gambar di mana wajah tidak dapat terdeteksi.
    """
    deleted_count = 0
    total_checked = 0

    print(f"--- Memulai Proses Pembersihan Dataset di Folder: {FACES_DIR} ---")

    if not os.path.exists(FACES_DIR):
        print(f"Error: Folder dataset '{FACES_DIR}' tidak ditemukan.")
        return

    # Iterasi ke setiap folder user
    for user_id in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, user_id)
        if not os.path.isdir(user_dir):
            continue

        print(f"\nMemeriksa folder untuk user: {user_id}")
        
        # Ambil daftar file gambar
        image_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            image_path = os.path.join(user_dir, filename)
            total_checked += 1
            
            # Cek deteksi wajah
            face = detect_and_crop(image_path)

            if face is None:
                # Jika tidak ada wajah, hapus file
                try:
                    print(f"  -> {filename} [GAGAL DETEKSI - MENGHAPUS FILE]")
                    os.remove(image_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"  -> {filename} [GAGAL MENGHAPUS: {e}]")
            else:
                print(f"  -> {filename} [OK]")

    print("\n--- Proses Pembersihan Selesai ---")
    print(f"Total file diperiksa: {total_checked}")
    print(f"Total file dihapus: {deleted_count}")

if __name__ == "__main__":
    # Konfirmasi sebelum menghapus untuk keamanan
    confirm = input(
        f"PERINGATAN: Skrip ini akan MENGHAPUS file gambar secara permanen dari '{FACES_DIR}' jika wajah tidak terdeteksi.\n"
        f"Apakah Anda yakin ingin melanjutkan? (y/n): "
    )
    if confirm.lower() == 'y':
        clean_dataset()
    else:
        print("Pembersihan dibatalkan oleh pengguna.")