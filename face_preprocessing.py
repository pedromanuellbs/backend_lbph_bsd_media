# face_preprocessing.py

import cv2
from deepface import DeepFace
import numpy as np

def detect_and_crop(img_path):
    """
    Detect a single face from the image at img_path using DeepFace's detector,
    crop, convert to grayscale, and return as a NumPy array.
    Returns None if no face is detected.
    """
    try:
        # DeepFace.extract_faces akan mendeteksi dan meng-crop wajah.
        # Backend 'mtcnn' digunakan secara default.
        face_obj = DeepFace.extract_faces(
            img_path=img_path, 
            enforce_detection=True, 
            detector_backend='mtcnn',
            align=True
        )
        
        # Ambil data gambar dari hasil
        face_img = face_obj[0]['face'] # Ambil wajah pertama yang terdeteksi
        
        # Konversi gambar dari float (0-1) ke uint8 (0-255)
        face_img_uint8 = (face_img * 255).astype(np.uint8)

        # Konversi ke Grayscale
        gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)
        
        return gray_face
        
    except Exception as e:
        print(f"Wajah tidak terdeteksi atau error di DeepFace: {e}")
        return None