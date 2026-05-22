import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Snake AI Lens")

st.title("🐍 Snake Species AI Detector")
st.write("Upload or take a photo of a snake")

# Load MobileNet model (lightweight pretrained AI)
@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

model = load_model()

# Image input
img_file = st.camera_input("Take a picture")
uploaded_file = st.file_uploader("Or upload image", type=["jpg","png","jpeg"])

image = None

if img_file:
    image = Image.open(img_file)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    # preprocess
    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # prediction
    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0]

    label = decoded[0][1]
    confidence = decoded[0][2]

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}")
