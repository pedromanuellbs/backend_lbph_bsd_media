# # Di file: face_preprocessing.py

# import cv2
# from deepface import DeepFace
# import numpy as np
# import os

# def detect_and_crop(img_path):
#     """
#     Detect a single face, crop, resize to 96x96, convert to grayscale,
#     and return as a NumPy array.
#     """
#     try:
#         # DeepFace akan mendeteksi dan meng-crop wajah
#         face_obj = DeepFace.extract_faces(
#             img_path=img_path,
#             enforce_detection=True,
#             detector_backend='mtcnn',
#             align=True
#         )

#         face_img = face_obj[0]['face']
#         face_img_uint8 = (face_img * 255).astype(np.uint8)
#         gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)

#         # Seragamkan ukuran semua wajah menjadi 96x96 piksel
#         resized_face = cv2.resize(gray_face, (96, 96))

#         return resized_face # Kembalikan gambar yang ukurannya sudah seragam

#     except Exception as e:
#         print(f"Wajah tidak terdeteksi atau error di DeepFace pada file {os.path.basename(img_path)}: {e}")
#         return None



# Di file: face_preprocessing.py

import os
import cv2
from deepface import DeepFace
import numpy as np

# def detect_and_crop(img_path):
#     """
#     Detect a single face, crop, resize to 96x96, convert to grayscale,
#     and return as a NumPy array.
#     Alignment is turned off to increase detection success rate.
#     """
#     try:
#         # DeepFace akan mendeteksi dan meng-crop wajah
#         face_obj = DeepFace.extract_faces(
#             img_path=img_path,
#             enforce_detection=True,
#             detector_backend='mtcnn',
#             align=False  # <-- PERUBAHAN UTAMA DI SINI
#         )
        
#         # Mengambil data gambar dari hasil deteksi pertama
#         face_img = face_obj[0]['face']
        
#         # Konversi gambar dari float (0-1) ke uint8 (0-255)
#         face_img_uint8 = (face_img * 255).astype(np.uint8)

#         # Konversi ke Grayscale untuk model LBPH
#         gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)

#         # Seragamkan ukuran semua wajah menjadi 96x96 piksel
#         resized_face = cv2.resize(gray_face, (96, 96))
        
#         return resized_face # Kembalikan gambar yang ukurannya sudah seragam

#     except Exception as e:
#         # Memberikan nama file saat terjadi error agar lebih informatif
#         print(f"Wajah tidak terdeteksi atau error di DeepFace pada file {os.path.basename(img_path)}: {e}")
#         return None


# Di file: face_preprocessing.py

# face_preprocessing.py

# import os
# import cv2
# from deepface import DeepFace
# import numpy as np

# def detect_and_crop(img_path):
#     """
#     Mendeteksi satu wajah dari gambar, memotong, mengubah ukuran menjadi 96x96,
#     mengonversi ke grayscale, dan mengembalikannya sebagai array NumPy.
    
#     Alignment dinonaktifkan untuk meningkatkan tingkat keberhasilan deteksi.
#     Menggunakan backend 'opencv' yang cepat.
#     """
#     try:
#         # DeepFace akan mendeteksi dan memotong wajah dari gambar
#         face_obj = DeepFace.extract_faces(
#             img_path=img_path,
#             enforce_detection=True,
#             detector_backend='opencv',
#             align=False
#         )
        
#         # Mengambil data gambar dari hasil deteksi pertama
#         # Hasil 'face' adalah array numpy dengan format (W, H, C) dan nilai float 0-1
#         face_img = face_obj[0]['face']
        
#         # Konversi gambar dari float (0-1) ke uint8 (0-255) yang dibutuhkan OpenCV
#         face_img_uint8 = (face_img * 255).astype(np.uint8)

#         # Konversi ke Grayscale untuk model LBPH
#         gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)

#         # Seragamkan ukuran semua wajah menjadi 96x96 piksel
#         resized_face = cv2.resize(gray_face, (96, 96))
        
#         return resized_face

#     except Exception as e:
#         # Memberikan nama file saat terjadi error agar lebih informatif
#         print(f"Wajah tidak terdeteksi atau error di DeepFace pada file {os.path.basename(img_path)}: {e}")
#         return None


import os
import cv2
from deepface import DeepFace
import numpy as np

def detect_and_crop(img_path):
    """
    Mendeteksi satu wajah dari gambar, memotong, mengubah ukuran menjadi 96x96,
    mengonversi ke grayscale, dan mengembalikannya sebagai array NumPy.
    
    Fungsi ini adalah langkah pre-processing untuk data latih model LBPH.
    """
    # Dapatkan nama file untuk logging
    filename = os.path.basename(img_path)
    
    try:
        # DeepFace akan mendeteksi dan memotong wajah dari gambar
        face_obj = DeepFace.extract_faces(
            img_path=img_path,
            enforce_detection=True,
            detector_backend='mtcnn',  # <-- Diubah sesuai permintaan
            align=False
        )
        
        face_img = face_obj[0]['face']
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        gray_face = cv2.cvtColor(face_img_uint8, cv2.COLOR_RGB2GRAY)
        resized_face = cv2.resize(gray_face, (96, 96))
        
        # <-- Log untuk deteksi yang berhasil ditambahkan
        print(f"[OK] Wajah terdeteksi di: {filename}")
        
        return resized_face

    except Exception as e:
        # <-- Log untuk deteksi yang gagal
        print(f"[GAGAL] Wajah tidak terdeteksi di: {filename}")
        return None