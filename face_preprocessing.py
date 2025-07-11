import cv2
import logging
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Inisialisasi sekali saja saat module di-load
mtcnn = MTCNN(image_size=96, margin=0)

def detect_and_crop(img_path):
    """Baca gambar, deteksi wajah dengan MTCNN, crop & kembalikan grayscale (96×96)."""
    try:
        logger.info(f"Processing image: {img_path}")
        
        # Try to open and read the image
        try:
            img = Image.open(img_path)
            logger.info(f"Successfully loaded image {img_path}, size: {img.size}")
        except Exception as e:
            logger.error(f"Read error - Could not load image {img_path}: {str(e)}")
            return None
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info(f"Converted image {img_path} to RGB mode")
        
        # Detect face using MTCNN
        try:
            logger.info(f"Running MTCNN face detection on {img_path}")
            face = mtcnn(img)  # Tensor (3,96,96) atau None
            
            if face is None:
                logger.error(f"Face detection error - No face detected in {img_path}")
                return None
            
            logger.info(f"MTCNN successfully detected face in {img_path}")
            
            # Konversi tensor ke array grayscale uint8
            if len(face.shape) == 3:
                # Face tensor is (3, 96, 96), convert to (96, 96, 3)
                face_array = face.permute(1, 2, 0).mul(255).byte().numpy()
            else:
                logger.error(f"MTCNN error - Unexpected face tensor shape: {face.shape}")
                return None
                
            # Convert to grayscale
            gray = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
            
            logger.info(f"Successfully processed {img_path} - Face detected and converted to grayscale (96x96)")
            return gray
            
        except Exception as e:
            logger.error(f"MTCNN error - Face detection failed for {img_path}: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"General error processing image {img_path}: {str(e)}")
        return None
