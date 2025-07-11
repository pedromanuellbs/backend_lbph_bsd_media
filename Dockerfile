# Gunakan image PyTorch CPU resmi
# Versi 1.13.1-cpu adalah salah satu yang paling stabil untuk Python 3.10
# Anda bisa coba juga 'pytorch/pytorch:2.2.2-cpu' jika Anda ingin versi lebih baru
FROM pytorch/pytorch:1.13.1-cpu

# Set direktori kerja di dalam container
WORKDIR /app

# Non-aktifkan buffering output Python untuk log real-time
ENV PYTHONUNBUFFERED 1

# Instal pustaka sistem yang mungkin masih dibutuhkan oleh OpenCV (jika tidak termasuk di base image)
# atau oleh paket Python lainnya yang akan diinstal.
# build-essential, cmake biasanya sudah ada di image 'devel' tapi mungkin tidak di 'runtime'
# libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender1: Ini penting untuk runtime OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Salin requirements.txt untuk paket-paket LAINNYA yang TIDAK termasuk dalam image base
# (misalnya Flask, gunicorn, scikit-learn, firebase-admin, google-api-python-client, facenet-pytorch)
COPY requirements.txt .

# Install sisa pustaka Python
# Hapus torch, torchvision, numpy, opencv-python, Pillow dari requirements.txt Anda
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file kode aplikasi Anda
COPY . .

# Perintah untuk menjalankan aplikasi
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]