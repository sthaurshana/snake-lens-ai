import streamlit as st
import numpy as np
from PIL import Image

st.title("🐍 Snake Species Detector (Debug Mode)")

camera = st.camera_input("Take a picture")
upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image)

    st.success("App is working")

    st.info("Prediction: Cobra (demo mode)")
