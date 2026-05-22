import streamlit as st
import numpy as np
from PIL import Image

st.title("🐍 Snake Detector (Demo Version)")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image)

    st.warning("Model not loaded on cloud (fixing deployment issue)")

    st.info("To enable prediction, run model locally OR use API deployment")
