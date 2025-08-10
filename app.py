import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

st.title("🐟 Multiclass Fish Image Classification")

# Model path
MODEL_PATH = "fish_model.h5"  # ✅ Corrected extension

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    file_id = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"  # Your Google Drive file ID
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_fish_model():
    return load_model(MODEL_PATH)

model = load_fish_model()

# ✅ Replace with your actual class labels
class_labels = [
    "Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", 
    "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp",
    "Striped Red Mullet", "Trout", "Sea Sprat", "Other"
]

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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
