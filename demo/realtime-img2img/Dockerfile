# ================================
# Stage 1: Build Frontend with Node
# ================================
FROM node:20-alpine AS frontend

# Install dependencies
RUN apk add --no-cache libc6-compat git

WORKDIR /app

# Clone the frontend repository
RUN git clone https://github.com/timosenko1/StreamDiffusionSC.git .

WORKDIR /app/demo/realtime-img2img/frontend

# Install Node dependencies and build
RUN npm ci
RUN npm run build

# ================================
# Stage 2: Setup Backend with CUDA 12.2
# ================================
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS backend

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_MAJOR=20

# Install system dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libgl1 \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy model files
COPY --chown=user:user models/u2net_human_seg.onnx /home/user/app/models/
COPY --chown=user:user models/dreamshaper_lcm.safetensors /home/user/app/models/dreamshaper_lcm/
COPY --chown=user:user models/last-000024.safetensors /home/user/app/models/

# Rename model file if necessary
RUN mv /home/user/app/models/dreamshaper_lcm/dreamshaper_lcm.safetensors /home/user/app/models/dreamshaper_lcm/model.safetensors

# Install pyenv and Python
RUN curl https://pyenv.run | bash

ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

ARG PYTHON_VERSION=3.10.12

RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

WORKDIR $HOME/app

# Copy built frontend from the frontend stage
COPY --from=frontend --chown=user:user /app .

WORKDIR $HOME/app/demo/realtime-img2img

# Install Python dependencies
RUN pip install --no-cache-dir -U -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/cumulo-autumn/StreamDiffusion.git@main#egg=streamdiffusion

# Install ONNXRuntime with GPU support
RUN pip install --no-cache-dir onnxruntime-gpu

# Install PyTorch compatible with CUDA 12.2
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu122 && \
    pip install --no-cache-dir torchaudio

# Alternatively, if Option 2 is preferred:
# RUN pip install --no-cache-dir torch torchvision torchaudio \
#     --index-url https://download.pytorch.org/whl/cu122 \
#     --extra-index-url https://pypi.org/simple

# Create and set cache directory
RUN mkdir -p /home/user/.cache/huggingface/transformers

ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# (Optional) Suppress FutureWarnings
# ENV PYTHONWARNINGS="ignore::FutureWarning"

# Create /data/hub directory with appropriate permissions
USER root
RUN mkdir -p /data/hub && chown -R user:user /data/hub
USER user

# Start the application
CMD ["python", "main.py", "--debug", "--acceleration", "xformers"]
