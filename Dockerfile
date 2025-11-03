# Use official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA="1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gnupg \
    build-essential \
    libglib2.0-0 \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    libgtk-3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install MongoDB
RUN curl -L https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-debian12-7.0.5.tgz -o /tmp/mongodb.tgz && \
    tar -xzf /tmp/mongodb.tgz -C /tmp/ && \
    cp /tmp/mongodb-linux-x86_64-debian12-7.0.5/bin/* /usr/local/bin/ && \
    rm -rf /tmp/mongodb* && \
    mkdir -p /data/db && \
    chmod -R 777 /data/db

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install MMDetection from source for better compatibility
RUN git clone https://github.com/open-mmlab/mmdetection.git /tmp/mmdetection && \
    cd /tmp/mmdetection && \
    pip install -v -e . && \
    rm -rf /tmp/mmdetection

# Create directories for the application
RUN mkdir -p /app/data /app/models /app/configs /app/results

# Set file permissions as per user preference
RUN chmod -R 666 /app

# Copy application code (when building)
# COPY . .

# Expose port for potential web services
EXPOSE 8000

# Default command
CMD ["python", "-c", "print('Car Damage Detection Environment Ready!')"]
