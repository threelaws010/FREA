FROM python:3.10-bullseye

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install torch with ROCm if available
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
RUN pip install easyocr transformers ultralytics streamlit  # Add others as needed

# Copy your app code
COPY . /app
WORKDIR /app

# Start your app
CMD ["python", "make_md_input_GPU.py"]
