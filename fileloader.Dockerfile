FROM python:3.10-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch-related packages from PyTorch CPU index
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other packages from PyPI (default)
RUN pip install --no-cache-dir \
    langchain \
    langchain-openai \
    langchain-neo4j \
    langchain-community \
    openai \
    flask \
    streamlit \
    transformers \
    pytesseract \
    opencv-python-headless \
    Pillow \
    numpy \
    ultralytics

# Copy app code
COPY . .

# Expose ports
EXPOSE 8507
EXPOSE 5000

# Run
CMD ["python", "fileloade.py"]
CMD ["streamlit", "run", "fileloade.py", "--server.port=8507"]
