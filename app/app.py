import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pathlib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Banana Leaf Disease Detection",
    page_icon="🍃",
    layout="centered"
)

# ==========================================
# MODEL
# ==========================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "banana_disease_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# ==========================================
# TITLE
# ==========================================

st.title("🍃 Banana Leaf Disease Detection")

st.write(
    "Upload a banana leaf image to classify it as "
    "**Sigatoka** or **Wilt**."
)

st.info(
    "Prototype model trained using MobileNetV2 "
    "with the available labeled dataset."
)

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a banana leaf image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Disease"):

        # Resize image
        resized_image = image.resize((224, 224))

        # Convert to NumPy array
        image_array = np.array(resized_image)

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # MobileNetV2 preprocessing
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )

        # Model prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )[0][0]

        # Class mapping
        if prediction >= 0.5:
            disease = "Wilt"
            confidence = prediction
        else:
            disease = "Sigatoka"
            confidence = 1 - prediction

        # ======================================
        # RESULT
        # ======================================

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Disease: {disease}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        # ======================================
        # INFORMATION
        # ======================================

        if disease == "Sigatoka":

            st.write(
                "**Detected class:** Sigatoka leaf spot"
            )

            st.write(
                "The prototype model classified the "
                "uploaded image as Sigatoka."
            )

        else:

            st.write(
                "**Detected class:** Wilt"
            )

            st.write(
                "The prototype model classified the "
                "uploaded image as Wilt."
            )

        st.warning(
            "This is a research prototype trained on a "
            "very small dataset. The prediction should "
            "not be treated as a definitive agricultural diagnosis."
        )

else:

    st.write(
        "👆 Upload an image to begin detection."
    )