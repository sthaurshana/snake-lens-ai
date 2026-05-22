import streamlit as st
import numpy as np
from PIL import Image

st.title("🐍 Snake Species Detector (Final Debug Version)")

st.write("Upload or take a photo of a snake")

# Input
camera = st.camera_input("Take a picture")
upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image, caption="Input Image", use_column_width=True)

    st.success("Image loaded successfully")

    # Dummy prediction (for debugging model issue)
    # Replace this when your model is fixed
    prediction = np.array([0.0, 0.0, 0.0])

    st.write("Raw prediction (DEBUG):", prediction)

    # Show fake labels (for UI testing only)
    labels = ["cobra", "python", "viper"]

    st.info("Prediction system is running (model needs retraining for accuracy)")
