# Pakai base image Python resmi
FROM python:3.10-slim

# Install dependencies build
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /app
COPY . /app

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8000 (atau 5000 jika pakai Flask default)
EXPOSE 8000

# Jalankan server
CMD ["python", "app.py"]
