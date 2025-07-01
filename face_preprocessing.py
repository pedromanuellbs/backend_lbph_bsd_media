import cv2
from facenet_pytorch import MTCNN
from PIL import Image

# Inisialisasi sekali saja saat module di-load
mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    """Baca gambar, deteksi wajah dengan MTCNN, crop & kembalikan grayscale (96×96)."""
    img = Image.open(img_path)
    face = mtcnn(img)  # Tensor (3,96,96) atau None
    if face is None:
        return None
    # Konversi tensor ke array grayscale uint8
    gray = face.permute(1, 2, 0).mul(255).byte().numpy()
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return gray
