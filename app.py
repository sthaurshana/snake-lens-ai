import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model (IMPORTANT: new file name)
model = tf.keras.models.load_model("model.keras")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.title("🐍 Snake Species Detector")

st.write("Upload or take a picture")

camera = st.camera_input("Take a picture")
upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image, caption="Input Image")

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    class_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))

    result = labels[class_index]

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
