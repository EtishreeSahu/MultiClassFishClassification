import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?id=1gNh7c-LdAew8x94y6VCa1I9rfTMgZGky"
MODEL_PATH = "mobilenetv2_fish_model.h5"

# Download model from Google Drive if not already present
@st.cache_resource
def load_fish_model():
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = load_fish_model()

# Class labels (update if needed)
class_labels = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
    'Red Mullet', 'Red Sea Bream', 'Sea Bass',
    'Shrimp', 'Striped Red Mullet', 'Trout',
    'Salmon', 'Other'
]

# Streamlit UI
st.title("🐟 Multiclass Fish Classification - MobileNetV2")
st.write("Upload a fish image to predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize without blurring
    img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)

    # Preprocess
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Output
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Class Probabilities:")
    for label, prob in zip(class_labels, preds[0]):
        st.write(f"{label}: {prob*100:.2f}%")
