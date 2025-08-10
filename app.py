import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# ---------------------------
# Streamlit App Title
# ---------------------------
st.title("🐟 Multiclass Fish Image Classification")

# ---------------------------
# Model File Setup
# ---------------------------
MODEL_PATH = "fish_model.keras"
   # Change to "fish_model.h5" if your model is in .h5 format
FILE_ID = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# ---------------------------
# Download Model if Missing
# ---------------------------
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive...")
    try:
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# ---------------------------
# Load Model with Caching
# ---------------------------
@st.cache_resource
def load_fish_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found after download.")
        st.stop()
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_fish_model()

# ---------------------------
# Class Labels
# ---------------------------
class_labels = [
    "Albacore Tuna", "Atlantic Mackerel", "Bigeye Tuna", "Black Sea Sprat", 
    "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", 
    "Sea Bass", "Shrimpfish", "Striped Red Mullet"
]  # Replace with your actual labels

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess image for prediction
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Display results
    st.success(f"✅ Predicted: **{predicted_class}** with confidence **{confidence:.2f}%**")

    # Show class probabilities
    st.subheader("📊 Prediction Probabilities")
    prob_dict = {class_labels[i]: float(preds[0][i]) for i in range(len(class_labels))}
    st.write(prob_dict)
