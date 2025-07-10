# 1. Gunakan base image python yang ringan
FROM python:3.10-slim

# 2. Install system dependencies, termasuk cmake dan python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

# 3. Tetapkan direktori kerja di dalam container
WORKDIR /app

# 4. Salin HANYA file requirements.txt terlebih dahulu
COPY requirements.txt .

# 5. Install semua Pustaka Python dari satu sumber
RUN pip install --no-cache-dir -r requirements.txt

# 6. Salin sisa kode aplikasi Anda
COPY . .

# 7. Expose port yang digunakan aplikasi
EXPOSE 8000

# 8. Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["python", "app.py"]