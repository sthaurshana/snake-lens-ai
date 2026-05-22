import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load model
model = load_model("model.h5")

classes = ["cobra", "python", "viper"]

st.title("🐍 Snake Species Detector (AI)")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image")

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    st.success(f"Prediction: {result}")
