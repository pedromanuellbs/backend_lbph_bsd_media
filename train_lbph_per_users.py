import os
import cv2
import numpy as np

DATASET_DIR = 'faces'
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

def train_lbph_for_user(user_id):
    user_dir = os.path.join(DATASET_DIR, str(user_id))
    images, labels = [], []
    label_map = {}
    label_counter = 0

    # Debug: tampilkan isi folder
    if not os.path.exists(user_dir):
        print(f"Folder {user_dir} tidak ditemukan!")
        return

    files = os.listdir(user_dir)
    print(f"Isi {user_dir}: {files}")

    for fname in files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            img_path = os.path.join(user_dir, fname)
            print("Cek file:", img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label_counter)
            else:
                print("GAGAL baca gambar:", img_path)

    if not images:
        print(f"Tidak ada gambar valid di {user_dir}, skip.")
        return

    label_map[label_counter] = str(user_id)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels))
    model.write(os.path.join(MODEL_DIR, f'{user_id}_lbph.yml'))
    np.save(os.path.join(MODEL_DIR, f'{user_id}_labels.npy'), label_map)
    print(f"Model untuk user {user_id} selesai ditrain dan disimpan.")

# Train untuk user 1 sampai 8
for user_id in range(1, 9):
    train_lbph_for_user(user_id)