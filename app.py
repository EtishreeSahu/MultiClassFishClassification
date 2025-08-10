import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

st.title("🐟 Multiclass Fish Image Classification")

# Model path
MODEL_PATH = "fish_model.keras"

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    file_id = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"  # Your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_fish_model():
    return load_model(MODEL_PATH)

model = load_fish_model()

# Class labels - update these with your actual fish class names
class_labels = ["Class1", "Class2", "Class3"]

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Predicted: **{predicted_class}** with confidence **{confidence:.2f}%**")
