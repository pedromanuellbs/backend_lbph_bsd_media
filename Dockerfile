# Gunakan base image Python 3.10 yang ringan dan stabil
FROM python:3.10-slim-buster

# Set direktori kerja di dalam container
WORKDIR /app

# Non-aktifkan buffering output Python untuk log real-time
ENV PYTHONUNBUFFERED 1

# Update package list dan install pustaka sistem yang dibutuhkan
# build-essential, cmake: Untuk kompilasi paket Python tertentu (ekstensi C/C++)
# libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender1: Dependensi runtime untuk OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    libopenblas-dev \
    liblapack-dev \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# --- START INSTALASI BERTahap ---

# Tahap 1: Instal dependensi dasar terlebih dahulu
# Ini adalah paket yang jarang bermasalah dan dibutuhkan oleh banyak lainnya.
RUN pip install --no-cache-dir \
    Flask==2.2.5 \
    gunicorn==22.0.0 \
    gevent==22.10.2 \
    requests==2.31.0 \
    firebase-admin==6.4.0 \
    google-api-python-client==2.128.0

# Tahap 2: Instal NumPy dan Pillow (seringkali pra-syarat untuk ML/CV)
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    Pillow==10.2.0

# Tahap 3: Instal PyTorch dan Torchvision (dengan index-url yang krusial)
# Ini adalah titik rawan utama. Jika gagal, errornya harus lebih spesifik untuk torch.
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Tahap 4: Instal OpenCV (titik rawan kedua)
# Tahap 4: Instal OpenCV Contrib
RUN pip install --no-cache-dir opencv-contrib-python
# ATAU (jika yang di atas masih gagal)
# RUN pip install --no-cache-dir opencv-contrib-python # Biarkan pip memilih versi terbaru

# --- PERUBAHAN DI SINI: Pisahkan dua paket ini ---
# Tahap 5: Instal Scikit-learn
RUN pip install --no-cache-dir \
    facenet-pytorch==2.6.0 \
    scikit-learn==1.1.3

# Tahap 6: Instal Facenet-PyTorch (Ini harusnya baik-baik saja jika scikit-learn berhasil)
RUN pip install --no-cache-dir \
    facenet-pytorch==2.5.2
# --- END PERUBAHAN ---

# Salin semua file kode aplikasi Anda ke dalam container
COPY . .

# =============================================================
# === UNDUH MODEL ML SAAT BUILD DARI FIREBASE STORAGE ===
# =============================================================
# GANTI "URL_PUBLIK_LBPH_MODEL_ANDA" dan "URL_PUBLIK_LABELS_MAP_ANDA"
# DENGAN URL ASLI DARI FIREBASE STORAGE ANDA
RUN wget -O /app/lbph_model.xml "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flbph_model.xml?alt=media&token=26656ed8-3cd1-4220-a07d-aad9aaeb91f5" && \
    wget -O /app/labels_map.txt "https://firebasestorage.googleapis.com/v0/b/db-ta-bsd-media.firebasestorage.app/o/face-recognition-models%2Flabels_map.txt?alt=media&token=2ab5957f-78b2-41b0-a1aa-b2f1b8675f54"
# =============================================================

# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]