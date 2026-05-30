import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Snake AI Lens")

st.title("🐍 Snake Species Detector")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Load labels
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

img_file = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload image", type=["jpg","png","jpeg"])

image = None

if img_file:
    image = Image.open(img_file)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image:
    image = image.convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    class_index = np.argmax(preds)
    label = class_names[class_index]
    confidence = float(np.max(preds))

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")
