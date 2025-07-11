# 1. Gunakan base image python yang ringan
FROM python:3.10-slim

# 2. Install system dependencies, ganti wget dengan curl
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
    curl \
    libcudnn8 \
    cuda-cudart-11-2 \
    cuda-libraries-11-2 \ # <-- TAMBAHKAN TANDA \ DI SINI
 && rm -rf /var/lib/apt/lists/*

# 3. Tetapkan direktori kerja di dalam container
WORKDIR /app

# 4. "Panggang" model SFace (menggunakan curl)
RUN mkdir -p /root/.deepface/weights \
 && curl -sL -o /root/.deepface/weights/sface_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/sface_weights.h5

# 5. Salin HANYA file requirements.txt terlebih dahulu untuk caching
COPY requirements.txt .

# 6. Install semua Pustaka Python dari satu sumber
RUN pip install --no-cache-dir -r requirements.txt

# 7. Salin sisa kode aplikasi Anda
COPY . .

# 8. Expose port yang digunakan aplikasi
EXPOSE 8000

# 9. Perintah untuk menjalankan aplikasi saat container dimulai
CMD ["python", "app.py"]