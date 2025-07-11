# Gunakan base image Python 3.10
# 'slim-buster' adalah pilihan yang baik karena lebih kecil tetapi masih berbasis Debian untuk apt-get
FROM python:3.10-slim-buster

# Set direktori kerja di dalam container
WORKDIR /app

# Non-aktifkan buffer output Python agar log langsung terlihat
ENV PYTHONUNBUFFERED 1

# Update package list dan install pustaka sistem yang dibutuhkan oleh OpenCV, PyTorch, dan build tools
# build-essential & cmake: Untuk kompilasi paket Python tertentu (misal, yang punya ekstensi C/C++)
# libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender1: Untuk dependensi runtime OpenCV
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
# Ini memastikan pip install hanya dijalankan ulang jika requirements.txt berubah
COPY requirements.txt .

# Install semua pustaka Python
# --no-cache-dir untuk menghindari caching pip di dalam container (mengurangi ukuran image)
# --break-system-packages (opsional, untuk Python 3.11+ jika ada masalah environment)
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file kode aplikasi Anda ke dalam container
# '.' berarti salin semua dari direktori konteks build (tempat Dockerfile berada) ke WORKDIR /app
COPY . .

# =============================================================
# === OPSIONAL: JIKA ANDA INGIN MENGUNDUH MODEL ML SAAT BUILD ===
# =============================================================
# Jika Anda telah melatih model secara lokal dan mengunggahnya ke Firebase Storage
# atau cloud storage lainnya, Anda bisa mengunduhnya di sini untuk mengurangi
# beban CPU/Memori saat aplikasi startup.
# Anda perlu mengganti "URL_PUBLIK_MODEL_ANDA" dan "URL_PUBLIK_LABELS_ANDA"
# dengan URL sebenarnya dari Firebase Storage.
#
# RUN wget -O /app/lbph_model.xml "URL_PUBLIK_MODEL_ANDA" && \
#     wget -O /app/labels_map.txt "URL_PUBLIK_LABELS_ANDA"
#
# Hati-hati: Jika model sangat besar, ini juga bisa menyebabkan timeout saat build.
# Jika Anda mengandalkan app.py untuk mendownload/melatih model saat runtime (melalui
# fungsi initial_model_check dan update_lbph_model_incrementally), maka baris ini
# tidak perlu ditambahkan atau biarkan tetap dikomentari.
# =============================================================

# Perintah yang akan dijalankan saat container dimulai
# Menggunakan Gunicorn sebagai WSGI server untuk Flask
# --worker-class gevent: Menggunakan Gevent untuk penanganan koneksi asinkron (lebih efisien)
# --timeout 120: Meningkatkan waktu timeout worker menjadi 120 detik (default 30-60 detik).
#                Ini penting jika initial_model_check atau operasi startup lain memakan waktu.
# -b 0.0.0.0:8080: Mengikat server ke semua antarmuka di port 8080.
#                 Pastikan port ini dibuka di konfigurasi Railway Anda.
CMD ["gunicorn", "--worker-class", "gevent", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]