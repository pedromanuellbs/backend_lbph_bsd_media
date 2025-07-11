#!/usr/bin/env python3
"""
Pre-warm DeepFace models during Docker build to avoid runtime downloads.
This script downloads and caches the necessary models for the application.
"""

import os
import sys
import numpy as np
from deepface import DeepFace

def prewarm_models():
    """Download and cache the necessary DeepFace models."""
    print("=== Starting DeepFace Model Pre-warming ===")
    
    try:
        # Create dummy face data for testing
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 1. Pre-warm SFace model (used in gdrive_match.py for face verification)
        print("Pre-warming SFace model...")
        try:
            result = DeepFace.verify(
                img1_path=dummy_face, 
                img2_path=dummy_face,
                model_name='SFace',
                enforce_detection=False
            )
            print("✓ SFace model loaded and cached successfully")
        except Exception as e:
            print(f"✗ Error pre-warming SFace model: {e}")
            return False
        
        # 2. Pre-warm face extraction with OpenCV backend (used in face_preprocessing.py)
        print("Pre-warming face extraction with OpenCV backend...")
        try:
            face_objs = DeepFace.extract_faces(
                img_path=dummy_face,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            print("✓ Face extraction with OpenCV backend loaded successfully")
        except Exception as e:
            print(f"✗ Error pre-warming face extraction: {e}")
            return False
        
        # 3. Check if models were downloaded to the expected directory
        deepface_dir = os.path.expanduser('~/.deepface')
        weights_dir = os.path.join(deepface_dir, 'weights')
        
        print(f"Checking model files in: {weights_dir}")
        if os.path.exists(weights_dir):
            model_files = os.listdir(weights_dir)
            print(f"Downloaded model files: {model_files}")
            
            # Check for SFace model specifically
            sface_model = 'face_recognition_sface_2021dec.onnx'
            if sface_model in model_files:
                print(f"✓ SFace model file found: {sface_model}")
            else:
                print(f"✗ SFace model file not found: {sface_model}")
                return False
        else:
            print(f"✗ Weights directory not found: {weights_dir}")
            return False
        
        print("=== DeepFace Model Pre-warming Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"✗ Fatal error during model pre-warming: {e}")
        return False

if __name__ == '__main__':
    success = prewarm_models()
    if not success:
        print("Model pre-warming failed!")
        sys.exit(1)
    else:
        print("Model pre-warming completed successfully!")
        sys.exit(0)