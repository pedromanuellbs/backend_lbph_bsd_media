# 1. Gunakan base image python yang ringan
FROM python:3.10-slim

# 2. Install system dependencies
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
 && rm -rf /var/lib/apt/lists/*

# 3. Tetapkan direktori kerja di dalam container
WORKDIR /app

# 4. TAMBAHKAN SWAP FILE UNTUK MEMORI CADANGAN
RUN fallocate -l 1G /swapfile \
 && chmod 600 /swapfile \
 && mkswap /swapfile \
 && swapon /swapfile

# 5. "Panggang" model SFace (ringan) ke dalam image
RUN mkdir -p /root/.deepface/weights \
 && curl -sL -o /root/.deepface/weights/sface_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/sface_weights.h5

# 6. Salin requirements.txt
COPY requirements.txt .

# 7. Install Pustaka Python
RUN pip install --no-cache-dir -r requirements.txt

# 8. Salin sisa kode
COPY . .

# 9. Expose port
EXPOSE 8000

# 10. Perintah untuk menjalankan aplikasi
CMD ["python", "app.py"]