# face_data.py

import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from face_preprocessing import detect_and_crop
from config import FACES_DIR, MODEL_PATH, LABEL_MAP


def build_dataset():
    """Bangun X (array gambar), y (label), dan label_map dari folder faces/."""
    X, y, label_map = [], [], {}
    cur_label = 0

    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue

        if uid not in label_map:
            label_map[uid] = cur_label
            cur_label += 1
        lbl = label_map[uid]

        for fn in os.listdir(user_dir):
            img_path = os.path.join(user_dir, fn)
            gray = detect_and_crop(img_path)
            if gray is None:
                continue
            X.append(gray)
            y.append(lbl)

    return np.array(X), np.array(y), label_map

def cross_validate_lbph(X, y, n_splits=10):
    """Lakukan 10-fold stratified CV, kembalikan metrik rata-rata."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, precs, recs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model = cv2.face.LBPHFaceRecognizer_create()
        # Latih
        model.train([X[i] for i in train_idx], y[train_idx])
        # Prediksi
        preds = [model.predict(X[i])[0] for i in val_idx]
        accs.append(accuracy_score(y[val_idx], preds))
        precs.append(precision_score(y[val_idx], preds, average='macro'))
        recs.append(recall_score(y[val_idx], preds, average='macro'))

    return {
        'accuracy': float(np.mean(accs)),
        'precision': float(np.mean(precs)),
        'recall': float(np.mean(recs))
    }

def train_and_evaluate():
    """
    1) Bangun dataset penuh
    2) Hitung 10-fold CV
    3) Latih model akhir di semua data
    4) Simpan model & label map
    5) Kembalikan dictionary metrik
    """
    X, y, label_map = build_dataset()
    if len(X) == 0:
        raise RuntimeError("Tidak ada data untuk dilatih")

    # 10-fold CV
    metrics = cross_validate_lbph(X, y)

    # Latih final model di semua data
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(list(X), y)
    model.write(MODEL_PATH)

    # Simpan label_map ke disk
    with open(LABEL_MAP, 'w') as f:
        for uid, lbl in label_map.items():
            f.write(f"{lbl}:{uid}\n")

    return metrics
