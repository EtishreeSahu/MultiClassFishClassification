import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

st.title("🐟 Multiclass Fish Image Classification")

# Path where model will be stored
MODEL_PATH = "fish_model.keras"

# Google Drive File ID for the model
FILE_ID = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"  # Replace with your file ID

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# Check if model file exists after download
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file could not be downloaded. Please check the Google Drive link and permissions.")
    st.stop()

@st.cache_resource
def load_fish_model():
    return load_model(MODEL_PATH)

model = load_fish_model()

# Class labels - replace with your actual fish class names
class_labels = [
    "Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", 
    "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", 
    "Trout", "Sea Smelt", "Striped Sea Bream"
]

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Predicted: **{predicted_class}** with confidence **{confidence:.2f}%**")
