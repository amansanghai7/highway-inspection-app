import os
import gdown
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(page_title="Highway Inspection AI", layout="centered")

# Google Drive file ID for best.pt
file_id = 'https://drive.google.com/file/d/1_-9yptXBigTm284Q6F09cg2Des08pb_0/view?usp=sharing'
model_path = 'best.pt'

# Download model from Google Drive if not exists
if not os.path.exists(model_path):
    st.info("Downloading YOLOv8 model from Google Drive...")
    url = f'https://drive.google.com/file/d/1_-9yptXBigTm284Q6F09cg2Des08pb_0/view?usp=sharing'
    gdown.download(url, model_path, quiet=False)

# Load the YOLO model
model = YOLO(model_path)

# UI layout
st.title("🚧 Highway Inspection and Maintenance using AI")
st.write("Upload an image to detect speed breakers, road markings, barriers, and dividers.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(Image.open(image_path), caption="Uploaded Image", use_container_width=True)

    # Run model prediction
    st.info("Running detection...")
    results = model.predict(image_path, save=True, conf=0.3)

    # Display result
    result_image_path = os.path.join(results[0].save_dir, os.path.basename(image_path))
    st.image(result_image_path, caption="Detection Result", use_container_width=True)
    st.success("Detection complete!")