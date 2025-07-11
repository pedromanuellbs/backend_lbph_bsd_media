# face_preprocessing.py

import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import cv2 # <--- TAMBAHKAN BARIS INI

# Inisialisasi MTCNN sekali saja untuk efisiensi
# Menggunakan 'cuda' jika tersedia, jika tidak 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, post_process=False, device=device)

def detect_and_crop(image_path):
    """
    Mendeteksi wajah dari path gambar, meng-crop, dan mengonversinya ke grayscale.
    Fungsi ini sekarang lebih tangguh terhadap file gambar yang rusak atau format aneh.
    """
    try:
        img = Image.open(image_path)
        # Pastikan gambar dikonversi ke format RGB
        img = img.convert('RGB')
    except Exception as e:
        print(f"ERROR: Gagal membuka atau mengonversi gambar {image_path}: {e}")
        return None

    try:
        # Tangkap RuntimeError dari MTCNN
        face = mtcnn(img)
        if face is None:
            return None # Tidak ada wajah yang terdeteksi
    except RuntimeError as e:
        print(f"ERROR: Gagal memproses gambar {image_path} dengan MTCNN: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan tak terduga saat deteksi wajah di {image_path}: {e}")
        return None

    # Mengubah tensor PyTorch ke format numpy array untuk OpenCV
    face_np = face.permute(1, 2, 0).to('cpu').numpy()
    # Konversi ke Grayscale untuk model LBPH
    gray_face = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
    
    return gray_face