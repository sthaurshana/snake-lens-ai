import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.title("🐍 Snake Species Identifier")

st.write("Take a photo or upload an image of a snake")

# Camera input
img_file = st.camera_input("Take a picture")

# Upload option
uploaded_file = st.file_uploader("Or upload image", type=["jpg", "png", "jpeg"])

image = None

if img_file:
    image = Image.open(img_file)
elif uploaded_file:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    result = labels[class_index]

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")