from facenet_pytorch import MTCNN
from PIL import Image
import cv2

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)
    if face is None:
        return None

    # Jika ada batch dim, buang
    if face.dim() == 4:
        face = face.squeeze(0)

    # face sekarang (3,96,96)
    rgb = face.mul(255).byte().permute(1, 2, 0).numpy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray
