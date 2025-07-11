# File: face_preprocessing.py
import os
import cv2
from deepface import DeepFace
import numpy as np

def detect_and_crop(img_data):
    """
    Mendeteksi wajah dari DATA GAMBAR (bukan path), crop, resize, dan konversi.
    Mengembalikan gambar grayscale 96x96 jika berhasil, atau None jika gagal.
    """
    try:
        # PAKAI MTCNN SAJA
        face_objs = DeepFace.extract_faces(
            img_path=img_data,
            detector_backend='mtcnn',
            enforce_detection=True,
            align=False
        )
        
        first_face = face_objs[0]
        face_img = first_face['face']
        
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)
        resized_face = cv2.resize(gray_face, (96, 96))
        
        return resized_face
        
    except Exception as e:
        return None