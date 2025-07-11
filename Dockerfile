# Gunakan image PyTorch CPU resmi yang sudah terintegrasi OpenCV atau base image yang mendukungnya
# pytorch/pytorch:1.13.1-cpu adalah contoh, tapi sering tidak ada
# Mari kita coba tag yang lebih umum dan pastikan itu versi CPU
# Coba salah satu dari ini, dimulai dari yang paling stabil/umum:
# Option A: Image PyTorch CPU yang lebih umum (jika tersedia dan mengandung Python 3.10)
FROM pytorch/pytorch:2.0.1-cpu-py310 # Atau versi sejenis jika ada untuk Python 3.10
# Option B: Image Python yang populer untuk ML (misal: Anaconda/Miniconda)
# FROM continuumio/miniconda3:latest
# Kemudian Anda harus menginstal paket dengan `conda install` atau `pip install`
# dengan `conda` untuk manajemen environment, ini akan mengubah struktur RUN pip install Anda.

# Untuk sekarang, mari kita asumsikan image PyTorch yang spesifik (jika ada):
# Jika '2.0.1-cpu-py310' tidak ada, coba '2.0.1-cpu' atau cari di Docker Hub
FROM pytorch/pytorch:2.0.1-cpu-py310 


# Set direktori kerja di dalam container
WORKDIR /app

# Non-aktifkan buffering output Python untuk log real-time
ENV PYTHONUNBUFFERED 1

# Instal pustaka sistem yang mungkin masih dibutuhkan
# Meskipun base image PyTorch sudah lengkap, ini sebagai jaring pengaman
# libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender1: Penting untuk runtime OpenCV
# build-essential, cmake: Berguna jika ada paket Python lain yang perlu kompilasi
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Salin requirements.txt untuk paket-paket LAINNYA
# yang TIDAK termasuk dalam image base (misalnya Flask, gunicorn, firebase-admin, dll.)
COPY requirements.txt .

# Install sisa pustaka Python
# Hapus torch, torchvision, numpy, opencv-python, Pillow dari requirements.txt Anda
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file kode aplikasi Anda
COPY . .

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]