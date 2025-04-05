import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolo import segment_and_classify_image

st.title("YOLO Segmentation and Classification")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL and convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Segmenting and classifying...")

    # Run segmentation and classification
    segments = segment_and_classify_image(image_np)

    for idx, (segment, label) in enumerate(segments):
        st.write(f"Segment {idx + 1}: {label}")
        st.image(segment, caption=f"{label} Segment {idx + 1}", use_column_width=True)
