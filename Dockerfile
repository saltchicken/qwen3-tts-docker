# ‼️ We use the official image matching your logs (PyTorch 2.4.0 + CUDA 12.4)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set the timezone and language (from your logs)
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set architecture for your 3080 (8.6) and 4090 (8.9)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"

# ‼️ CRITICAL: Force the compiler to use the Old ABI to match this PyTorch image
# This prevents the "undefined symbol" error you saw earlier.
ENV CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

# Install build dependencies (git, ninja)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    build-essential \
    sox \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Flash Attention 2
# We use --no-build-isolation to ensure it uses the pre-installed PyTorch
ENV MAX_JOBS=4
RUN pip install flash-attn==2.6.3 --no-build-isolation

RUN pip install -U qwen-tts
RUN pip install fastapi uvicorn

# Set the workspace
WORKDIR /workspace
CMD ["python", "server.py"]
