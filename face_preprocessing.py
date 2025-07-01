from facenet_pytorch import MTCNN
from PIL import Image
import cv2

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    """
    Deteksi wajah dari img_path → crop + resize 96×96 → 
    kembalikan grayscale numpy array atau None.
    """
    img = Image.open(img_path)
    face = mtcnn(img)  # bisa jadi Tensor[3,96,96] atau Tensor[1,3,96,96]
    if face is None:
        return None

    # Jika ada batch dim (1,3,96,96), buang jadi (3,96,96)
    if face.dim() == 4:
        face = face.squeeze(0)

    # Konversi ke uint8 numpy dan grayscale
    # face sekarang pasti shape (3,96,96)
    rgb = face.mul(255).byte().permute(1, 2, 0).numpy()  # (96,96,3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)         # (96,96)
    return gray
