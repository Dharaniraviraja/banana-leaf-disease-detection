import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "model/banana_disease_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 2. IMAGE TO PREDICTION
# ==========================================

def predict_disease(image_path):

    # Load image
    image = load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to array
    image_array = img_to_array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    # Our class mapping:
    # Sigatoka = 0
    # Wilt = 1

    if prediction >= 0.5:
        disease = "Wilt"
        confidence = prediction
    else:
        disease = "Sigatoka"
        confidence = 1 - prediction

    return disease, confidence


# ==========================================
# 3. ENTER IMAGE PATH
# ==========================================

image_path = input("Enter the image path: ")

disease, confidence = predict_disease(image_path)

print("\n======================================")
print("       BANANA LEAF PREDICTION")
print("======================================")

print("Prediction :", disease)
print("Confidence :", f"{confidence * 100:.2f}%")