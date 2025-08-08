import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import gdown

# Google Drive model link (replace with your MobileNetV2 .h5 link)
MODEL_URL = "https://drive.google.com/file/d/1gNh7c-LdAew8x94y6VCa1I9rfTMgZGky/view?usp=drive_link"
MODEL_PATH = "mobilenetv2_fish_model.h5"

# Download model from Google Drive
@st.cache_resource
def download_and_load_model():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

# Load the model
model = download_and_load_model()

# Class labels (adjust if your dataset classes are different)
class_labels = [
    "Black Sea Sprat", "Gilt-Head Bream", "Horse Mackerel", "Red Mullet",
    "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet",
    "Trout", "Salmon", "Tuna"
]

# Streamlit UI
st.title("🐟 Multiclass Fish Classification (MobileNetV2)")
st.write("Upload a fish image and the model will predict its species.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Resize using high-quality LANCZOS
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Preprocess for MobileNetV2
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Display results
    st.markdown(f"### Prediction: **{pred_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities:")
    for label, prob in zip(class_labels, preds):
        st.write(f"{label}: {prob * 100:.2f}%")
