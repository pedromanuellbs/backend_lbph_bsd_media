import os
import shutil

# Path sumber dan tujuan
source_dir = r"C:\Users\maspe\Downloads\archive (1)\Faces\Faces"
target_dir = r"D:\coding-files\ta-pace\tugas-akhir2\bsd_media_backend\faces"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for filename in os.listdir(source_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    if "_" in filename:
        user = filename.rsplit("_", 1)[0]
    else:
        user = os.path.splitext(filename)[0]

    user_folder = os.path.join(target_dir, user)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    src = os.path.join(source_dir, filename)
    dst = os.path.join(user_folder, filename)
    shutil.move(src, dst)

print("Semua file telah berhasil dipindahkan dan diorganisasi ke folder masing-masing user di path tujuan.")