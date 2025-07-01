FROM python:3.10-slim

# 1) Install system-deps untuk OpenCV, Pillow, dan PyTorch CPU
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

# 2) Copy hanya requirements.txt dulu untuk leverage Docker cache
COPY requirements.txt .

# 3) Upgrade pip & install Python deps (termasuk facenet-pytorch, torch, torchvision, scikit-learn, Pillow, dsb)
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# 4) Baru copy seluruh source
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
