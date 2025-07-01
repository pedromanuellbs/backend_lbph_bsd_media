FROM python:3.10-slim

# 1) Install system‐deps untuk OpenCV, Pillow, dan PyTorch CPU
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

# 2) Copy hanya requirements.txt dulu
COPY requirements.txt .

# 3) (Debug) Tampilkan isi requirements agar kita yakin sudah bersih
RUN echo "--- requirements.txt @ build time ---" \
 && cat requirements.txt

# 4) Upgrade pip & install Python deps
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# 5) Copy seluruh source code
COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
