# Base image with ROCm support for PyTorch
FROM rocm/dev-ubuntu-20.04:5.7-complete


# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Use python3.10 explicitly for pip installs
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 && \
    python3.10 -m pip install --no-cache-dir \
    langchain \
    langchain-community \
    openai \
    neo4j \
    yolov5 \
    pytesseract \
    pillow \
    opencv-python \
    transformers \
    unstructured \
    tiktoken \
    streamlit


# Copy your source code into the container
COPY . .

# Download YOLOv5s model if needed (optional)
 RUN python3 -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)"

# Default command to run your script
CMD ["python3", "fileloade.py"]
