
# 🐟 Multiclass Fish Image Classification

A deep learning project that classifies fish images into multiple categories using **MobileNetV2** and **Streamlit**.  
The model is trained on fish datasets and deployed as a web application for easy use.

---

## 🌐 Live Demo
Check out the live Streamlit app here:  
[Multiclass Fish Classification App](https://multiclassfishclassification-bmoodjp45evwcaxf2anwdk.streamlit.app/)

---

## 📂 Project Structure
```

multiclassfishclassification/
│
├── app.py                  # Streamlit app script
├── requirements.txt        # Python dependencies
├── mobilenetv2\_fish\_model.h5  # Trained model (loaded from Google Drive in app)
└── README.md               # Project documentation

````

---

## ⚙️ How It Works
1. Upload a fish image in JPG, JPEG, or PNG format.
2. The model preprocesses the image and resizes it to **224x224**.
3. The image is passed to a trained **MobileNetV2** model.
4. The app displays the predicted fish species and confidence score.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **TensorFlow / Keras**
- **Streamlit**
- **MobileNetV2 (Transfer Learning)**
- **gdown** (to download the model from Google Drive)
- **Pillow** (image handling)
- **NumPy**

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/etishreesahu/multiclassfishclassification.git

# Navigate to project directory
cd multiclassfishclassification

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
````

---

## 📌 Features

* Upload and classify fish images instantly
* MobileNetV2 transfer learning model
* Displays prediction confidence
* Lightweight and fast web app

---

## 👩‍💻 Author

**Etishree Sahu**
GitHub: [etishreesahu](https://github.com/etishreesahu)

```

This will display correctly on GitHub — **Author** will no longer be stuck in the code block.  

Do you also want me to add a **"Sample Predictions"** section with images for your README so it looks more appealing? That would make it more professional.
```
