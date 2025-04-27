FROM python:3.10-bullseye

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0 libgl1-mesa-glx wget git && apt-get clean

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python easyocr python-dotenv torch torchvision torchaudio transformers accelerate

EXPOSE 8501

CMD ["python3", "make_md_input_GPU.py"]

