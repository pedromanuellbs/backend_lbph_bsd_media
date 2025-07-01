from facenet_pytorch import MTCNN
from PIL import Image
import cv2

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)           # bisa None, atau Tensor

    if face is None:
        return None

    # Jika batch dim: (1, C, H, W) → (C, H, W)
    if face.dim() == 4:
        face = face.squeeze(0)

    # Jika cuma 1 channel (grayscale), duplikasi ke 3 channel
    if face.size(0) == 1:
        face = face.expand(3, -1, -1)

    # Pastikan shape sekarang (3,96,96)
    rgb = face.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray
