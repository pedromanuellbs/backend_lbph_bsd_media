from facenet_pytorch import MTCNN
import cv2
import torch
from PIL import Image
import numpy as np

mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    img = Image.open(img_path)
    face = mtcnn(img)           # bisa (3,96,96) atau (1,3,96,96) atau None
    if face is None:
        return None

    # kalau ada batch dim pertama, drop:
    if isinstance(face, torch.Tensor) and face.ndim == 4:
        face = face.squeeze(0)

    # sekarang face.ndim == 3: (C,H,W)
    gray = face.permute(1,2,0).mul(255).byte().numpy()  
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return gray
