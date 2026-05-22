import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

# Load labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

st.title("🐍 Snake Species Detector")

st.write("Upload or take a picture of a snake")

# Camera + upload
camera = st.camera_input("Take a picture")
upload = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

file = camera if camera else upload

if file:
    image = Image.open(file)
    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess (VERY IMPORTANT)
    img = image.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    st.write("Raw prediction:", prediction)  # DEBUG LINE

    class_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    if confidence < 0.5:
        result = "Unknown"
    else:
        result = labels[class_index]

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
