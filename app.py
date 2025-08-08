from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1gNh7c-LdAew8x94y6VCa1I9rfTMgZGky"
MODEL_PATH = "mobilenetv2_weights.h5"

# Download weights
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Rebuild architecture
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Load weights
model.load_weights(MODEL_PATH)
