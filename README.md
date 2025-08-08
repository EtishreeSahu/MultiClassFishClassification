# 🐟 Multiclass Fish Image Classification

A deep learning project that classifies fish images into 11 species using **MobileNetV2 (Transfer Learning)**.  
The best model is deployed via a **Streamlit web app** for real-time fish species prediction from user-uploaded images.

---

## 📌 Project Overview
This project leverages computer vision and transfer learning to identify fish species from images.  
We trained two models:
1. **CNN from Scratch** – Baseline model  
2. **MobileNetV2** – Pre-trained on ImageNet, fine-tuned for our dataset  

**Best Model:** MobileNetV2 with:
- **Accuracy:** 98.62%
- **Precision:** 0.9864
- **Recall:** 0.9862
- **F1-Score:** 0.9850

---

## 🗂 Dataset
- **Source:** Provided dataset of 11 fish species
- **Structure:** `train/`, `val/`, and `test/` folders
- **Size:** ~10K images

---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **gdown** (to load model from Google Drive)
- **NumPy, Pillow** (image preprocessing)

---

## 🚀 Deployment
The app is deployed on **Streamlit Cloud**.  
It downloads the trained `.h5` model from Google Drive at runtime.

🔗 **Live Demo:** [Click here to view the app](YOUR_STREAMLIT_APP_LINK) *(Replace with your link after deployment)*

---

## 📥 How to Run Locally
1. Clone this repo:
```bash
git clone https://github.com/YOUR_USERNAME/fish-image-classification-streamlit.git
cd fish-image-classification-streamlit
