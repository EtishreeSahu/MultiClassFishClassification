import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"
MODEL_PATH = "mobilenet_fish_model.h5"
CLASS_LABELS = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet',
    'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet',
    'Trout', 'Sea Sprat', 'Other'
]

# ------------------------------
# DOWNLOAD MODEL
# ------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... please wait.")
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code != 200:
            st.error("Failed to download the model. Please check the URL or permissions.")
            st.stop()
        open(MODEL_PATH, 'wb').write(r.content)
        st.success("Model downloaded successfully!")

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_fish_model():
    download_model()
    model = load_model(MODEL_PATH)
    return model

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_image(img, model):
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the model will classify it.")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_file = Image.open(uploaded_file).convert("RGB")
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    model = load_fish_model()

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            predicted_class, confidence = predict_image(image_file, model)
        st.success(f"Prediction: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
