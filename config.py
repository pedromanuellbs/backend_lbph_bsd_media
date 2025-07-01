# config.py
import os

# folder wajah, model, dan map label
BASE_DIR    = os.getcwd()
FACES_DIR   = os.path.join(BASE_DIR, "faces")
MODEL_PATH  = os.path.join(BASE_DIR, "lbph_model.xml")
LABEL_MAP   = os.path.join(BASE_DIR, "labels_map.txt")
