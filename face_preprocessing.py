# face_preprocessing.py (Versi Ringan dengan Haar Cascade)

import cv2
import os

# Muat model Haar Cascade untuk deteksi wajah dari file yang disediakan oleh OpenCV
# Pastikan opencv-python terinstall dengan benar
try:
    # Path ini biasanya bekerja jika OpenCV terinstall dengan baik
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise IOError(f"Tidak bisa memuat Haar Cascade dari path: {cascade_path}")
except Exception as e:
    print(f"Error saat memuat Haar Cascade: {e}")
    # Fallback atau hentikan aplikasi jika cascade tidak bisa dimuat
    face_cascade = None


def detect_and_crop(img_path):
    """
    Detect a single face dari gambar menggunakan Haar Cascade,
    crop & resize ke 96x96, konversi ke grayscale, dan kembalikan sebagai array NumPy.
    Return None jika wajah tidak terdeteksi.
    """
    if face_cascade is None:
        print("Haar Cascade tidak ter-load, deteksi wajah dibatalkan.")
        return None

    # Baca gambar menggunakan OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Gagal membaca gambar dari path: {img_path}")
        return None

    # Konversi ke grayscale untuk deteksi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    # scaleFactor dan minNeighbors bisa disesuaikan untuk akurasi vs kecepatan
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None # Tidak ada wajah yang terdeteksi

    # Ambil wajah pertama yang terdeteksi
    (x, y, w, h) = faces[0]
    
    # Crop wajah dari gambar grayscale
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize ke ukuran yang konsisten (96x96) seperti sebelumnya
    resized_face = cv2.resize(face_roi, (96, 96), interpolation=cv2.INTER_LINEAR)
    
    return resized_face