from facenet_pytorch import MTCNN
from PIL import Image
import cv2

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)

    if face is None:
        return None

    # Handle batch dimension
    if face.dim() == 4:
        face = face.squeeze(0)

    # Handle grayscale
    if face.size(0) == 1:
        face = face.expand(3, -1, -1)

    # Pastikan tensor 3 dimensi sebelum permute
    if face.dim() == 3:
        rgb = face.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return gray
    else:
        raise ValueError(f"Tensor shape tidak sesuai: {face.shape}")