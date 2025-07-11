# Gunakan base image Python 3.10
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Non-aktifkan buffer output Python
ENV PYTHONUNBUFFERED 1

# Update dan install pustaka sistem yang dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Install semua pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file kode aplikasi Anda ke dalam container
COPY . .

# =============================================================
# === BARIS BARU: JALANKAN SCRIPT UNTUK DOWNLOAD MODEL ML ===
# =============================================================
# Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]
