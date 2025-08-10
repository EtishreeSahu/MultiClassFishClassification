@st.cache_resource
def load_fish_model():
    import requests
    file_id = "1mpaj_mwcshSinDIHmWdHcc-S-xdPcUvq"
    model_url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "mobilenet_fish_weights.h5"
    
    if not os.path.exists(model_path):
        r = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(r.content)
    
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(11, activation='softmax')
    ])

    model.load_weights(model_path)
    return model
