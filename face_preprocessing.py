# Di file: face_preprocessing.py

import cv2
from deepface import DeepFace
import numpy as np
import os

def detect_and_crop(img_path):
    """
    Detect a single face, crop, resize to 96x96, convert to grayscale,
    and return as a NumPy array.
    """
    try:
        # DeepFace akan mendeteksi dan meng-crop wajah
        face_obj = DeepFace.extract_faces(
            img_path=img_path,
            enforce_detection=True,
            detector_backend='mtcnn',
            align=True
        )

        face_img = face_obj[0]['face']
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)

        # Seragamkan ukuran semua wajah menjadi 96x96 piksel
        resized_face = cv2.resize(gray_face, (96, 96))

        return resized_face # Kembalikan gambar yang ukurannya sudah seragam

    except Exception as e:
        print(f"Wajah tidak terdeteksi atau error di DeepFace pada file {os.path.basename(img_path)}: {e}")
        return None