import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

st.title("🐟 Multiclass Fish Image Classification")

# Google Drive model download settings
MODEL_PATH = "fish_model.keras"
FILE_ID = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"  # Your file ID from Google Drive
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Download model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    try:
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# Function to load the model
@st.cache_resource
def load_fish_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please check your Google Drive link and permissions.")
        st.stop()
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_fish_model()

# Class labels - replace with your actual fish species names
class_labels = [
    "Class1", "Class2", "Class3",
    "Class4", "Class5", "Class6",
    "Class7", "Class8", "Class9",
    "Class10", "Class11"
]

# File uploader
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Predicted: **{predicted_class}** with confidence **{confidence:.2f}%**")
