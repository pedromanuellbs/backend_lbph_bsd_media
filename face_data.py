# import os
# import numpy as np
# import cv2
# import torch
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
# # Struktur ini memudahkan pelatihan (train) dan pengujian (test) model face recognition.

# import numpy as np
# import cv2
# import torch
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
#     model = cv2.face.LBPHFaceRecognizer_create()
#     model.train(list(X), y)
#     model.write(MODEL_PATH)

#     # Simpan label_map
#     with open(LABEL_MAP, 'w') as f:
#         for uid, lbl in label_map.items():
#             f.write(f"{lbl}:{uid}\n")

#     return metrics



#pake ini jadi gacor anjj
import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from face_preprocessing import detect_and_crop
# Pastikan file config.py Anda sudah ada dan berisi variabel yang diperlukan
# from config import FACES_DIR, MODEL_PATH, LABEL_MAP

# --- Placeholder untuk config.py jika tidak ada ---
FACES_DIR = r"D:\coding-files\ta-pace\tugas-akhir2\bsd_media_backend\faces"
MODEL_PATH = "lbph_face_model.yml"
LABEL_MAP = "lbph_label_map.txt"
# ---------------------------------------------

def build_dataset():
    """
    Baca semua gambar di FACES_DIR, crop/grayscale, kembalikan X, y, label_map.
    """
    X, y, label_map = [], [], {}
    cur_label = 0

    if not os.path.exists(FACES_DIR):
        print(f"Error: Direktori dataset tidak ditemukan di {FACES_DIR}")
        return np.array([]), np.array([]), {}

    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue

        if uid not in label_map:
            label_map[uid] = cur_label
            cur_label += 1
        lbl = label_map[uid]

        for fn in os.listdir(user_dir):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(user_dir, fn)
                gray = detect_and_crop(img_path)
                if gray is None:
                    print(f"Wajah tidak terdeteksi di {img_path}")
                    continue
                X.append(gray)
                y.append(lbl)

    return np.array(X), np.array(y), label_map

def cross_validate_lbph(X, y, n_splits=10):
    """
    Lakukan stratified K-Fold CV, kembalikan metrik rata-rata.
    (Fungsi ini sudah ada dan tidak diubah)
    """
    n_samples = len(X)
    if n_samples < 2:
        return {'accuracy': 0, 'precision': 0, 'recall': 0}

    splits = min(n_splits, n_samples)
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accs, precs, recs = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        model = cv2.face.LBPHFaceRecognizer_create()
        # Perlu diubah menjadi list of arrays untuk training
        X_train_fold = [X[i] for i in train_idx]
        y_train_fold = y[train_idx]
        model.train(X_train_fold, y_train_fold)
        
        preds = [model.predict(X[i])[0] for i in val_idx]
        
        # zero_division=0 untuk menangani kasus di mana tidak ada prediksi untuk kelas tertentu
        accs.append(accuracy_score(y[val_idx], preds))
        precs.append(precision_score(y[val_idx], preds, average='macro', zero_division=0))
        recs.append(recall_score(y[val_idx], preds, average='macro', zero_division=0))

    return {
        'accuracy': float(np.mean(accs)),
        'precision': float(np.mean(precs)),
        'recall': float(np.mean(recs))
    }

def evaluate_with_train_test_split(X, y):
    """
    (FUNGSI BARU)
    Evaluasi model menggunakan pembagian 80% latih dan 20% uji.
    Ini sesuai dengan bagian 3.1.3.2 di proposal Anda.
    """
    if len(np.unique(y)) < 2:
         # Tidak bisa melakukan stratified split jika hanya ada 1 kelas
        return {'accuracy': None, 'precision': None, 'recall': None, 'error': 'Not enough classes for stratified split'}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nUkuran data latih (80%): {len(X_train)}")
    print(f"Ukuran data uji (20%): {len(X_test)}")
    
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(list(X_train), y_train)
    
    preds = [model.predict(face)[0] for face in X_test]
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='macro', zero_division=0)
    recall = recall_score(y_test, preds, average='macro', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall)
    }


def train_and_evaluate():
    """
    (FUNGSI UTAMA YANG DIMODIFIKASI)
    Bangun dataset, lakukan kedua jenis evaluasi, latih model final, dan simpan.
    """
    print("Membangun dataset...")
    X, y, label_map = build_dataset()
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Tidak ada data wajah untuk dilatih")
    
    print(f"\nTotal {n_samples} sampel wajah ditemukan dari {len(label_map)} individu.")

    # 1. Evaluasi dengan 10-Fold Cross-Validation (sesuai bagian 3.1.3.4)
    print("\n--- Mengevaluasi dengan 10-Fold Cross-Validation ---")
    cv_metrics = cross_validate_lbph(X, y)
    print(f"Metrik Rata-rata CV: {cv_metrics}")

    # 2. Evaluasi dengan 80/20 Train-Test Split (sesuai bagian 3.1.3.2)
    print("\n--- Mengevaluasi dengan 80/20 Train-Test Split ---")
    split_metrics = evaluate_with_train_test_split(X, y)
    print(f"Metrik 80/20 Split: {split_metrics}")

    # Latih model akhir dengan SEMUA data untuk production
    print("\nMelatih model final menggunakan seluruh dataset...")
    final_model = cv2.face.LBPHFaceRecognizer_create()
    final_model.train(list(X), y)
    final_model.write(MODEL_PATH)
    print(f"Model final telah disimpan di: {MODEL_PATH}")

    # Simpan label_map
    with open(LABEL_MAP, 'w') as f:
        for uid, lbl in label_map.items():
            f.write(f"{lbl}:{uid}\n")
    print(f"Label map telah disimpan di: {LABEL_MAP}")

    return {
        "cross_validation_metrics": cv_metrics,
        "train_test_split_metrics": split_metrics
    }

# Untuk menjalankan file ini secara langsung
if __name__ == "__main__":
    all_metrics = train_and_evaluate()
    print("\n================ HASIL AKHIR ================")
    print(f"Metrik dari Cross-Validation: {all_metrics['cross_validation_metrics']}")
    print(f"Metrik dari Train-Test Split: {all_metrics['train_test_split_metrics']}")
    print("===========================================")