from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import torch

# Inisialisasi MTCNN untuk deteksi wajah
# device='cpu' memastikan proses berjalan di CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=96, margin=0, device=device)

def detect_and_crop(img_path):
    """
    Deteksi satu wajah dari gambar menggunakan MTCNN, crop & resize ke 96x96,
    konversi ke grayscale, dan kembalikan sebagai array NumPy.
    Return None jika wajah tidak terdeteksi.
    """
    try:
        # Buka gambar dan pastikan formatnya RGB
        img = Image.open(img_path).convert("RGB")
    except IOError:
        print(f"Gagal membuka atau membaca file gambar: {img_path}")
        return None

    # Deteksi dan crop wajah; mengembalikan tensor
    face = mtcnn(img)
    
    if face is None:
        return None

    # Konversi tensor PyTorch ke format gambar yang bisa diproses OpenCV
    # (C,H,W) -> (H,W,C) -> BGR -> Grayscale
    rgb_face = face.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    gray_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
    
    return gray_face
