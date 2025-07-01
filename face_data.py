import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from face_preprocessing import detect_and_crop
from config import FACES_DIR  # atau pindahkan konstanta FACES_DIR ke modul utils jika suka

def build_dataset():
    X, y, label_map = [], [], {}
    cur = 0
    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue
        if uid not in label_map:
            label_map[uid] = cur
            cur += 1
        lbl = label_map[uid]
        for fn in os.listdir(user_dir):
            path = os.path.join(user_dir, fn)
            gray = detect_and_crop(path)
            if gray is None:
                continue
            X.append(gray)
            y.append(lbl)
    return np.array(X), np.array(y), label_map

# tambahkan juga fungsi split, cross‐validation, train_and_evaluate() di sini
