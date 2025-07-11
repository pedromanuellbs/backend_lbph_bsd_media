# import os
# import numpy as np
# import cv2
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from face_preprocessing import detect_and_crop
# from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# # Dataset Structure:
# # faces/
# #   <user_id>/
# #     img_front.jpg      # menghadap depan
# #     img_left.jpg       # menoleh kiri
# #     img_right.jpg      # menoleh kanan
# #     img_up.jpg         # menghadap atas
# #     img_down.jpg       # menghadap bawah
# #
# # Setiap folder user_id berisi variasi posisi wajah (.jpg).
#good
# # Struktur ini memudahkan pelatihan (train) dan pengujian (test) model face recognition.

# import numpy as np
# import cv2
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from face_preprocessing import detect_and_crop
# from config import FACES_DIR, MODEL_PATH, LABEL_MAP


# def build_dataset():
#     """
#     Baca semua gambar di FACES_DIR, crop/grayscale, kembalikan X, y, label_map.
#     """
#     X, y, label_map = [], [], {}
#     cur_label = 0

#     for uid in os.listdir(FACES_DIR):
#         user_dir = os.path.join(FACES_DIR, uid)
#         if not os.path.isdir(user_dir):
#             continue

#         if uid not in label_map:
#             label_map[uid] = cur_label
#             cur_label += 1
#         lbl = label_map[uid]

#         for fn in os.listdir(user_dir):
#             img_path = os.path.join(user_dir, fn)
#             gray = detect_and_crop(img_path)
#             if gray is None:
#                 continue
#             X.append(gray)
#             y.append(lbl)

#     return np.array(X), np.array(y), label_map


# def cross_validate_lbph(X, y, n_splits=10):
#     """
#     Lakukan stratified K-Fold CV, kembalikan metrik rata-rata.
#     Jika sampel kurang dari n_splits, gunakan n_splits = len(X).
#     """
#     n_samples = len(X)
#     if n_samples < 2:
#         # Tidak cukup data untuk CV
#         return {'accuracy': None, 'precision': None, 'recall': None}

#     # Atur n_splits sesuai jumlah sampel
#     splits = min(n_splits, n_samples)
#     skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
#     accs, precs, recs = [], [], []

#     for train_idx, val_idx in skf.split(X, y):
#         model = cv2.face.LBPHFaceRecognizer_create()
#         model.train([X[i] for i in train_idx], y[train_idx])
#         preds = [model.predict(X[i])[0] for i in val_idx]
#         accs.append(accuracy_score(y[val_idx], preds))
#         precs.append(precision_score(y[val_idx], preds, average='macro'))
#         recs.append(recall_score(y[val_idx], preds, average='macro'))

#     return {
#         'accuracy': float(np.mean(accs)),
#         'precision': float(np.mean(precs)),
#         'recall': float(np.mean(recs))
#     }


# def train_and_evaluate():
#     """
#     Bangun dataset, cross-validate jika memungkinkan, latih model final, simpan model dan label_map.
#     Kembalikan metrik CV (None jika tidak ada).
#     """
#     X, y, label_map = build_dataset()
#     n_samples = len(X)
#     if n_samples == 0:
#         raise RuntimeError("Tidak ada data wajah untuk dilatih")

#     # Cross-validation
#     metrics = cross_validate_lbph(X, y)

#     # Latih model akhir dengan semua data
#     model = cv2.face.LBPHFaceRecognizer_create(

#         radius=2,
#         neighbors=16,
#         grid_x=8,
#         grid_y=8,
#         threshold=100
#     )
#     model.train(list(X), y)
#     model.write(MODEL_PATH)

#     # Simpan label_map
#     with open(LABEL_MAP, 'w') as f:
#         for uid, lbl in label_map.items():
#             f.write(f"{lbl}:{uid}\n")

#     return metrics

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from face_preprocessing import detect_and_crop

# Constants
FACES_DIR = "faces"
MODEL_PATH = "lbph_model.xml"
LABELS_MAP_PATH = "labels_map.txt"
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from face_preprocessing import detect_and_crop
from config import FACES_DIR, MODEL_PATH, LABEL_MAP

def build_dataset():
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


def train_and_evaluate():
    """Train LBPH model and evaluate it, return metrics"""
    X, y, label_map = build_dataset()
    
    if len(X) == 0:
        return {'error': 'No training data available'}
    
    # Create reverse mapping for saving
    reverse_map = {v: k for k, v in label_map.items()}
    
    # Split data for evaluation
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
    
    # Train LBPH model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X_train, y_train)
    
    # Evaluate on test set
    predictions = []
    confidences = []
    
    for i, face in enumerate(X_test):
        label, confidence = model.predict(face)
        predictions.append(label)
        confidences.append(confidence)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    avg_confidence = np.mean(confidences)
    
    # Save model and labels
    model.save(MODEL_PATH)
    with open(LABELS_MAP_PATH, 'w') as f:
        for label, user_id in reverse_map.items():
            f.write(f"{label}:{user_id}\n")
    
    return {
        'accuracy': float(accuracy),
        'avg_confidence': float(avg_confidence),
        'samples_trained': len(X_train),
        'samples_tested': len(X_test),
        'users_count': len(label_map)
    }
def cross_validate_lbph(X, y, n_splits=10):
    n_samples = len(X)
    if n_samples < 2:
        return {'accuracy': None, 'precision': None, 'recall': None}

    splits = min(n_splits, n_samples)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accs, precs, recs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=16,
            grid_x=8,
            grid_y=8,
            threshold=100
        )
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
    X, y, label_map = build_dataset()
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Tidak ada data wajah untuk dilatih")

    metrics = cross_validate_lbph(X, y)

    # Konfigurasi optimal LBPH
    model = cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=16,
        grid_x=8,
        grid_y=8,
        threshold=100
    )
    model.train(list(X), y)
    model.write(MODEL_PATH)

    with open(LABEL_MAP, 'w') as f:
        for uid, lbl in label_map.items():
            f.write(f"{lbl}:{uid}\n")

    return metrics