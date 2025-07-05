# 1. Base image ringan
FROM python:3.10-slim

# 2. Install system deps untuk OpenCV headless
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# 3. Set direktori kerja
WORKDIR /app

# 4. Copy file requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy sisa kode aplikasi
COPY . .

# 6. Expose port yang digunakan oleh Railway
EXPOSE 8080

# 7. Jalankan aplikasi
CMD ["python", "app.py"]
