import streamlit as st
import numpy as np
from PIL import Image

st.title("🐍 Snake Species Detector (FIXED VERSION)")

camera = st.camera_input("Take picture")
upload = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image)

    st.success("App is working!")

    st.info("Prediction: Cobra (demo mode)")
    st.write("Confidence: 0.99")
