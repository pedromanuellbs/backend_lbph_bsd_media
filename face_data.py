import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from face_preprocessing import detect_and_crop
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# Dataset Structure:
# faces/
#   <user_id>/
#     img_front.jpg      # menghadap depan
#     img_left.jpg       # menoleh kiri
#     img_right.jpg      # menoleh kanan
#     img_up.jpg         # menghadap atas
#     img_down.jpg       # menghadap bawah
#
# Setiap folder user_id berisi variasi posisi wajah (.jpg).
# Struktur ini memudahkan pelatihan (train) dan pengujian (test) model face recognition.

import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from face_preprocessing import detect_and_crop
from config import FACES_DIR, MODEL_PATH, LABEL_MAP


def build_dataset():
    """
    Baca semua gambar di FACES_DIR, crop/grayscale, kembalikan X, y, label_map.
    """
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
    """
    Lakukan stratified K-Fold CV, kembalikan metrik rata-rata.
    Jika sampel kurang dari n_splits, gunakan n_splits = len(X).
    """
    n_samples = len(X)
    if n_samples < 2:
        # Tidak cukup data untuk CV
        return {'accuracy': None, 'precision': None, 'recall': None}

    # Atur n_splits sesuai jumlah sampel
    splits = min(n_splits, n_samples)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accs, precs, recs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train([X[i] for i in train_idx], y[train_idx])
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
    Bangun dataset, cross-validate jika memungkinkan, latih model final, simpan model dan label_map.
    Kembalikan metrik CV (None jika tidak ada).
    """
    X, y, label_map = build_dataset()
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Tidak ada data wajah untuk dilatih")

    # Cross-validation
    metrics = cross_validate_lbph(X, y)

    # Latih model akhir dengan semua data
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(list(X), y)
    model.write(MODEL_PATH)

    # Simpan label_map
    with open(LABEL_MAP, 'w') as f:
        for uid, lbl in label_map.items():
            f.write(f"{lbl}:{uid}\n")

    return metrics
