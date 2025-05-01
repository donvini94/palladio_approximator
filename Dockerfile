FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
# Note: Installing PyTorch with specific CUDA version
RUN pip3 install --no-cache-dir \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    tqdm==4.66.1 \
    seaborn==0.13.1 \
    matplotlib==3.8.2 \
    mlflow==2.10.0 \
    transformers==4.37.2 \
    tiktoken==0.5.2 \
    && pip3 install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the application code
COPY . /app/

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["bash"]
