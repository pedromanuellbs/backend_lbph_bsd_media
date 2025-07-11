import os
import numpy as np
import cv2
import torch # Diperlukan karena facenet_pytorch adalah dependency utama
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from config import FACES_DIR, MODEL_PATH, LABEL_MAP # Pastikan ini mengarah ke file config Anda
from face_preprocessing import detect_and_crop # Pastikan ini diimpor

def build_dataset():
    """
    Membaca semua gambar di FACES_DIR, melakukan crop/grayscale, dan mengembalikan X, y, label_map.
    Ini digunakan untuk pelatihan penuh atau saat model belum ada.
    """
    X, y, label_map = [], [], {}
    cur_label = 0

    # Memuat label_map yang sudah ada jika ada untuk menjaga konsistensi label
    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP, 'r') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    lbl_str, uid = parts
                    label_map[uid] = int(lbl_str)
            if label_map: # Mencari nomor label berikutnya yang tersedia
                cur_label = max(label_map.values()) + 1
        print(f"DEBUG: Initial label_map loaded: {label_map}")
        print(f"DEBUG: Starting cur_label for new users: {cur_label}")


    for uid in os.listdir(FACES_DIR):
        user_dir = os.path.join(FACES_DIR, uid)
        if not os.path.isdir(user_dir):
            continue

        if uid not in label_map:
            label_map[uid] = cur_label
            cur_label += 1
            print(f"DEBUG: New user '{uid}' assigned label: {label_map[uid]}")
        lbl = label_map[uid]

        for fn in os.listdir(user_dir):
            img_path = os.path.join(user_dir, fn)
            # Lewati jika bukan file gambar (misalnya, .DS_Store)
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue

            gray = detect_and_crop(img_path)
            if gray is None:
                print(f"WARNING: Gagal mendeteksi/crop wajah di {img_path}. Melompati.")
                continue
            X.append(gray)
            y.append(lbl)
    
    print(f"DEBUG: Dataset built with {len(X)} samples from {len(label_map)} users.")
    return np.array(X, dtype=object), np.array(y), label_map

def cross_validate_lbph(X, y, n_splits=10):
    """
    Melakukan stratified K-Fold CV, mengembalikan metrik rata-rata.
    """
    n_samples = len(X)
    if n_samples < 2:
        print("WARNING: Tidak cukup data untuk CV (<2 sampel).")
        return {'accuracy': None, 'precision': None, 'recall': None}

    splits = min(n_splits, n_samples)
    
    unique_classes = np.unique(y)
    if len(unique_classes) < splits:
        print(f"WARNING: Jumlah kelas unik ({len(unique_classes)}) kurang dari n_splits ({splits}). Mengurangi n_splits.")
        splits = max(1, len(unique_classes))
        if splits == 1 and len(unique_classes) == 1:
            print("WARNING: Hanya satu kelas unik. CV tidak relevan, metrik mungkin 1.0 atau 0.0.")
            return {'accuracy': None, 'precision': None, 'recall': None}

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accs, precs, recs = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = [X[i] for i in train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = [X[i] for i in val_idx]
        y_val_fold = y[val_idx]

        if not X_train_fold or not X_val_fold:
            print(f"WARNING: Fold {fold_idx + 1} memiliki set train/val kosong. Melewati fold ini.")
            continue

        model = cv2.face.LBPHFaceRecognizer_create()
        try:
            model.train(X_train_fold, y_train_fold)
        except Exception as e:
            print(f"ERROR: Gagal melatih model di fold {fold_idx + 1}: {e}")
            continue

        preds = []
        for img_idx, img_val in enumerate(X_val_fold):
            try:
                label, conf = model.predict(img_val)
                preds.append(label)
            except Exception as e:
                print(f"WARNING: Gagal prediksi gambar di fold {fold_idx + 1}, index {img_idx}: {e}")
                pass

        if not preds:
            print(f"WARNING: Tidak ada prediksi yang dibuat di fold {fold_idx + 1}. Melewati metrik.")
            continue

        y_val_list = y_val_fold.tolist()
        
        if len(y_val_list) > 0 and len(preds) > 0:
            unique_y_val = np.unique(y_val_list)
            unique_preds = np.unique(preds)
            
            if len(unique_y_val) <= 1:
                accs.append(accuracy_score(y_val_list, preds))
                precs.append(precision_score(y_val_list, preds, average='weighted', zero_division=0))
                recs.append(recall_score(y_val_list, preds, average='weighted', zero_division=0))
            else:
                accs.append(accuracy_score(y_val_list, preds))
                precs.append(precision_score(y_val_list, preds, average='macro', zero_division=0))
                recs.append(recall_score(y_val_list, preds, average='macro', zero_division=0))
        else:
            print(f"WARNING: Validation or prediction set is empty for fold {fold_idx + 1}. Skipping metrics for this fold.")

    return {
        'accuracy': float(np.mean(accs)) if accs else None,
        'precision': float(np.mean(precs)) if precs else None,
        'recall': float(np.mean(recs)) if recs else None
    }


def train_and_evaluate_full_dataset():
    """
    Membangun dataset dari awal, melakukan cross-validation jika memungkinkan, melatih model final, menyimpan model dan label_map.
    Mengembalikan metrik CV (None jika tidak ada).
    Fungsi ini untuk rebuild model secara total.
    """
    print("Membangun dataset dari awal untuk pelatihan penuh...")
    X, y, label_map = build_dataset()
    n_samples = len(X)
    if n_samples == 0:
        raise RuntimeError("Tidak ada data wajah untuk dilatih")

    print("Melakukan cross-validation...")
    metrics = cross_validate_lbph(X, y)
    print(f"Metrik CV: {metrics}")

    print("Melatih model akhir dengan semua data...")
    model = cv2.face.LBPHFaceRecognizer_create()
    try:
        model.train(list(X), y)
        model.write(MODEL_PATH)
        print(f"Model disimpan ke: {MODEL_PATH}")

        print(f"Menyimpan label_map ke: {LABEL_MAP}")
        with open(LABEL_MAP, 'w') as f:
            for uid, lbl in label_map.items():
                f.write(f"{lbl}:{uid}\n")
        print("Label map berhasil disimpan.")
    except Exception as e:
        print(f"ERROR: Gagal melatih atau menyimpan model/label_map: {e}")
        raise

    return metrics

def load_model_and_labels():
    """
    Memuat model LBPH dan label_map (lbl→user_id) dari filesystem.
    Mengembalikan tuple (model, label_map).
    Jika belum ada model atau error, mengembalikan (None, {}).
    """
    print("DEBUG: load_model_and_labels START")
    model = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: load_model_and_labels: MODEL_PATH {MODEL_PATH} TIDAK DITEMUKAN.")
        return None, {}
    if not os.path.exists(LABEL_MAP):
        print(f"ERROR: load_model_and_labels: LABEL_MAP {LABEL_MAP} TIDAK DITEMUKAN.")
        return None, {}

    try:
        model.read(MODEL_PATH)
        print("DEBUG: load_model_and_labels: Model berhasil dibaca.")
    except cv2.error as e:
        print(f"CRITICAL ERROR: load_model_and_labels: Error membaca model LBPH: {e}. Model mungkin corrupt atau tidak ada. Mengembalikan None.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print(f"INFO: Menghapus file model yang corrupt: {MODEL_PATH}")
        return None, {}
    except Exception as e:
        print(f"CRITICAL ERROR: load_model_and_labels: Error tak terduga memuat model: {e}. Mengembalikan None.")
        return None, {}

    label_map_loaded = {}
    try:
        with open(LABEL_MAP, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    lbl, uid = parts
                    label_map_loaded[int(lbl)] = uid
        print("DEBUG: load_model_and_labels: Label map berhasil dimuat.")
    except Exception as e:
        print(f"CRITICAL ERROR: load_model_and_labels: Gagal memuat label map dari '{LABEL_MAP}': {e}. Mengembalikan None.")
        if os.path.exists(LABEL_MAP):
            os.remove(LABEL_MAP)
            print(f"INFO: Menghapus file label map yang corrupt: {LABEL_MAP}")
        return None, {}

    reverse_label_map = {v: k for k, v in label_map_loaded.items()}
    print("DEBUG: load_model_and_labels END: BERHASIL.")
    return model, reverse_label_map


def update_lbph_model_incrementally(new_face_image_path, user_id):
    """
    Memuat model yang sudah ada, menambahkan wajah baru ke dalamnya, dan menyimpan kembali.
    Mengembalikan True jika update berhasil, False jika gagal.
    """
    print(f"DEBUG: Mencoba update model LBPH secara incremental untuk user: {user_id}")
    model, label_map = load_model_and_labels()

    if model is None or not label_map:
        print("INFO: Model belum ada atau label_map kosong. Melakukan pelatihan penuh untuk inisialisasi.")
        try:
            train_and_evaluate_full_dataset()
            model, label_map = load_model_and_labels()
            if model is None or not label_map:
                print("ERROR: Gagal membuat model setelah pelatihan penuh pertama.")
                return False
            print("INFO: Model awal berhasil dibuat.")
        except RuntimeError as e:
            print(f"ERROR: Gagal melakukan pelatihan penuh saat registrasi pertama: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Terjadi error tidak terduga saat inisialisasi model: {e}")
            return False

    print(f"DEBUG: Deteksi dan crop wajah dari: {new_face_image_path}")
    gray_face = detect_and_crop(new_face_image_path)
    if gray_face is None:
        print("ERROR: Gagal mendeteksi/meng-crop wajah baru dari gambar yang diberikan.")
        return False

    current_max_label = max(label_map.values()) if label_map else -1
    if user_id not in label_map:
        new_label = current_max_label + 1
        label_map[user_id] = new_label
        print(f"DEBUG: User '{user_id}' adalah baru, diberi label: {new_label}")
    else:
        new_label = label_map[user_id]
        print(f"DEBUG: User '{user_id}' sudah ada, menggunakan label: {new_label}")

    try:
        model.update([gray_face], np.array([new_label]))
        print("DEBUG: Model berhasil diupdate secara incremental.")
        model.write(MODEL_PATH)
        print(f"DEBUG: Model yang diupdate disimpan ke: {MODEL_PATH}")

        save_label_map = {v: k for k, v in label_map.items()}
        with open(LABEL_MAP, 'w') as f:
            for lbl, uid in save_label_map.items():
                f.write(f"{lbl}:{uid}\n")
        print(f"DEBUG: Label map yang diperbarui disimpan ke: {LABEL_MAP}")
        return True
    except Exception as e:
        print(f"ERROR: Error saat mengupdate model secara incremental atau menyimpan: {e}")
        import traceback
        traceback.print_exc()
        return False

# Untuk menjalankan file ini secara langsung (hanya untuk pengujian lokal)
if __name__ == "__main__":
    # Ini akan menjalankan fungsi train_and_evaluate_full_dataset()
    # yang akan melatih model dari semua data di FACES_DIR.
    # Pastikan FACES_DIR, MODEL_PATH, LABEL_MAP di config.py sudah benar.
    try:
        all_metrics = train_and_evaluate_full_dataset()
        print("\n================ HASIL AKHIR ================")
        print(f"Metrik dari Cross-Validation: {all_metrics['cross_validation_metrics']}")
        # train_and_evaluate_full_dataset tidak mengembalikan split_metrics, jadi ini mungkin None
        # print(f"Metrik dari Train-Test Split: {all_metrics['train_test_split_metrics']}")
        print("===========================================")
    except Exception as e:
        print(f"ERROR saat menjalankan train_model.py secara langsung: {e}")
        import traceback
        traceback.print_exc()
