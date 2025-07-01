# 1. Base image ringan
FROM python:3.10-slim

# 2. Install system deps untuk OpenCV headless, Pillow, dan kebutuhan build
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libjpeg-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy file requirements.txt lebih dulu
COPY requirements.txt .

# 4. Copy kode aplikasi
COPY . .

# 5. Upgrade pip, install CPU-only PyTorch & deps, lalu install sisa requirements
RUN pip install --upgrade pip \
  && pip install torch==2.2.2+cpu torchvision==0.17.2+cpu \
       --extra-index-url https://download.pytorch.org/whl/cpu \
  && pip install -r requirements.txt

# 6. Expose port dan jalankan aplikasi
EXPOSE 8000
CMD ["python", "app.py"]
