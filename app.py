import streamlit as st
import numpy as np
from PIL import Image

# Load labels from labels.txt
def load_labels(path="labels.txt"):
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# Dummy prediction function (replace with your ML model later)
def predict_snake(image, labels):
    # For now, just pick a random label
    return np.random.choice(labels)

# Streamlit UI
st.title("🐍 Snake Detector")

camera = st.camera_input("Take a picture")
upload = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image, caption="Input Image", use_column_width=True)

    st.success("App is working")

    labels = load_labels()
    prediction = predict_snake(image, labels)

    st.info(f"Prediction: {prediction}")
