import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

classes = ["cobra", "python", "viper"]

st.title("🐍 Snake Species Detector")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    result = classes[np.argmax(pred)]

    st.success(result)
