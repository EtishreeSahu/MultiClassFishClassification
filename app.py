import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Google Drive file ID
MODEL_FILE_ID = "1gNh7c-LdAew8x94y6VCa1I9rfTMgZGky"
MODEL_PATH = "mobilenet_fish_model.h5"

# Download model from Google Drive if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_fish_model():
    model = load_model(MODEL_PATH)
    return model

model = load_fish_model()

# Class labels
class_labels = [
    'Black Sea Sprat',
    'Gilt-Head Bream',
    'Horse Mackerel',
    'Red Mullet',
    'Red Sea Bream',
    'Sea Bass',
    'Shrimp',
    'Striped Red Mullet',
    'Trout',
    'Herring',
    'Salmon'
]

# App title
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish, and the model will predict its species.")

# File uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0

    # Ensure correct shape (1, 224, 224, 3)
    if img_array.shape != (224, 224, 3):
        st.error(f"Unexpected image shape {img_array.shape}, please upload a standard RGB image.")
    else:
        img_array = np.expand_dims(img_array, axis=0)

        # Debug shape
        st.write(f"Image array shape for prediction: {img_array.shape}")

        # Prediction
        preds = model.predict(img_array)
        pred_class = class_labels[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # Display results
        st.success(f"Prediction: **{pred_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Show class probabilities
        st.subheader("Class Probabilities:")
        for label, prob in zip(class_labels, preds[0]):
            st.write(f"{label}: {prob*100:.2f}%")
