import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import numpy as np
import gdown
import os

st.title("🐟 Multiclass Fish Image Classification")

# Google Drive file download
MODEL_PATH = "mobilenetv2_fish_model.h5"
if not os.path.exists(MODEL_PATH):
    file_id = "1yn-uC6GAfmcKdPWxFoVx5BvlRNoeFMxf"
gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_fish_model():
    # Rebuild the exact architecture
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(11, activation='softmax')  # Change 11 if your number of classes differs
    ])

    model.load_weights(MODEL_PATH)
    return model

model = load_fish_model()

# Replace with your actual fish class names
class_labels = [
    "Black Sea Sprat", "Gilt-Head Bream", "Horse Mackerel", "Red Mullet",
    "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout",
    "Salmon", "Tuna"
]

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.success(f"Predicted: **{predicted_class}** with confidence **{confidence:.2f}%**")

    # Show probabilities
    st.subheader("Class Probabilities:")
    for label, prob in zip(class_labels, preds[0]):
        st.write(f"{label}: {prob*100:.2f}%")
