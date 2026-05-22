import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Snake Lens AI")

st.title("🐍 Snake Species Identifier")
st.write("Upload or capture image of a snake")

# Camera + Upload
img_file = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload image", type=["jpg", "png", "jpeg"])

image = None

if img_file:
    image = Image.open(img_file)

elif uploaded_file:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    st.info("⚠️ Cloud version does not run TensorFlow model yet.")

    st.success("Prediction: (model disabled in cloud)")
    st.write("To enable AI, use local version or upgrade model later.")
