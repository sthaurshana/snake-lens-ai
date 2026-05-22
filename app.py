import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load model
model = load_model("model.h5")

# Class labels (CHANGE if your dataset is different)
classes = ["cobra", "python", "viper"]

# UI
st.title("🐍 Snake Species Detector (AI)")
st.write("Upload an image of a snake and get prediction")

# Upload image
file = st.file_uploader("Upload Snake Image", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess image
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
