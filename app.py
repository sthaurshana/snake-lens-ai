import streamlit as st
import numpy as np
from PIL import Image

st.title("🐍 Snake Species Detector")

st.write("Take a photo or upload an image")

# CAMERA INPUT (THIS IS THE FIX)
camera_file = st.camera_input("Take a picture")

upload_file = st.file_uploader("Or upload image", type=["jpg", "jpeg", "png"])

file = camera_file if camera_file else upload_file

if file:
    image = Image.open(file)
    st.image(image, caption="Input Image")

    st.success("Image received!")

    st.info("Prediction: Cobra (demo mode)")
