import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import cv2

# Inisialisasi MTCNN sekali saja untuk efisiensi dan pemilihan device
# Menggunakan 'cuda' jika tersedia, jika tidak 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"DEBUG: Menggunakan device: {device}")

# Inisialisasi MTCNN dengan parameter yang diinginkan
# image_size=96 dan margin=0 adalah penting untuk konsistensi input LBPH
# keep_all=False: Hanya mendeteksi wajah terbesar
# post_process=True: Menerapkan standardisasi gambar (prewhiten), direkomendasikan untuk FaceNet/MTCNN
mtcnn = MTCNN(image_size=96, margin=0, keep_all=False, post_process=True, device=device)

def detect_and_crop(image_path):
    """
    Mendeteksi wajah dari path gambar, meng-crop, dan mengonversinya ke grayscale.
    Fungsi ini sekarang lebih tangguh terhadap file gambar yang rusak atau format aneh.
    """
    print(f"DEBUG: detect_and_crop: Memulai pemrosesan gambar: {image_path}")
    try:
        img = Image.open(image_path)
        # Pastikan gambar dikonversi ke format RGB
        img = img.convert('RGB')
        print(f"DEBUG: detect_and_crop: Gambar berhasil dibuka. Format: {img.mode}, Ukuran: {img.size}")
    except Exception as e:
        print(f"ERROR: detect_and_crop: Gagal membuka atau mengonversi gambar {image_path}: {e}")
        return None

    try:
        # Deteksi wajah dengan MTCNN
        face = mtcnn(img)
        if face is None:
            print(f"ERROR: detect_and_crop: MTCNN GAGAL mendeteksi wajah di {image_path}. (face is None)")
            return None # Tidak ada wajah yang terdeteksi
        
        print(f"DEBUG: detect_and_crop: Wajah terdeteksi oleh MTCNN. Bentuk tensor: {face.shape if hasattr(face, 'shape') else 'N/A'}")

        # Mengubah tensor PyTorch ke format numpy array untuk OpenCV
        # Pastikan tensor dipindahkan ke CPU sebelum konversi ke NumPy
        face_np = face.permute(1, 2, 0).to('cpu').numpy()
        print(f"DEBUG: detect_and_crop: Tensor wajah dikonversi ke NumPy RGB. Bentuk: {face_np.shape}")

        # Konversi ke Grayscale untuk model LBPH
        gray_face = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        print(f"DEBUG: detect_and_crop: Wajah berhasil dikonversi ke grayscale. Bentuk: {gray_face.shape}")
        
        return gray_face

    except RuntimeError as e:
        print(f"ERROR: detect_and_crop: Gagal memproses gambar {image_path} dengan MTCNN (RuntimeError): {e}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: detect_and_crop: Terjadi kesalahan tak terduga saat deteksi wajah di {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
