from facenet_pytorch import MTCNN
from PIL import Image
import cv2

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    """
    Deteksi wajah → crop & resize 96×96 → kembalikan grayscale numpy array.
    """
    img = Image.open(img_path)
    face = mtcnn(img)  # bisa return Tensor dengan shape (3,96,96) atau (1,3,96,96)
    if face is None:
        return None

    # Buang batch dimension kalau ada
    if face.dim() == 4:
        face = face.squeeze(0)    # from (1,3,96,96) to (3,96,96)

    # face sekarang pasti shape (3,96,96)
    # Konversi ke (96,96,3) uint8 lalu ke grayscale
    rgb = face.mul(255).byte().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray
