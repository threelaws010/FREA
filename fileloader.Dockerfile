# Base image with ROCm support for PyTorch
FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.10_pytorch_2.1.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    tesseract-ocr \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python \
    pytesseract \
    pillow \
    yolov5 \
    langchain \
    neo4j \
    openai \
    transformers \
    unstructured \
    tiktoken

# Copy your source code into the container
COPY . .

# Download YOLOv5s model if needed (optional)
# RUN python3 -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)"

# Default command to run your script
CMD ["python", "your_script_name.py"]
