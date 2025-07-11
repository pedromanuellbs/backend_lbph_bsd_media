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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Salin file requirements.txt terlebih dahulu untuk memanfaatkan Docker layer caching
COPY requirements.txt .

# Install semua pustaka Python dari requirements.txt
# --no-cache-dir untuk menghindari caching pip di dalam container (mengurangi ukuran image)
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file kode aplikasi Anda ke dalam container
COPY . .

# Perintah yang akan dijalankan saat container dimulai
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]