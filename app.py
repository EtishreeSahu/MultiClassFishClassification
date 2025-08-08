import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Direct link to your MobileNetV2 model
MODEL_URL = "https://drive.google.com/uc?id=1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"
MODEL_PATH = "mobilenetv2_fish_model.h5"

@st.cache_resource
def load_fish_model():
    # Download and load the model
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_fish_model()

# Replace these with your actual fish species names
class_labels = [
    'Black Sea Sprat', 'Gilt-Head Bream', 'Horse Mackerel', 'Red Mullet',
    'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet',
    'Trout', 'Herring', 'Salmon'
]

st.title("🐟 Multiclass Fish Classification (MobileNetV₂)")
st.write("Upload a fish image and see which species the model predicts.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preserve image quality while resizing
    img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    
    st.write(f"### Prediction: **{pred_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    st.subheader("Class Probabilities:")
    for label, prob in zip(class_labels, preds):
        st.write(f"{label}: {prob*100:.2f}%")
