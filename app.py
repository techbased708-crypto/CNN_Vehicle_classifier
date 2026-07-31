import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# ── Model Load ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model = keras.models.load_model("cars_cnn.keras")
    return model

model = load_model()

# ── Class Names ──────────────────────────────────────────
# Kaggle dataset: mmohaiminulislam/vehicles-image-dataset
# Alphabetical order (same as TensorFlow reads folders)
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Truck', 'Van']

# ── UI ───────────────────────────────────────────────────
st.title("🚗 Vehicle Image Classifier")
st.markdown("CNN model se koi bhi vehicle image upload karo aur result dekho!")
st.markdown("---")

uploaded_file = st.file_uploader(
    "📂 Image upload karo (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing..."):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    # Result
    st.markdown("---")
    st.success(f"## 🎯 Result: **{predicted_class}**")
    st.info(f"### Confidence: **{confidence:.2f}%**")

    # All class probabilities
    st.markdown("### 📊 All Predictions:")
    for i, name in enumerate(CLASS_NAMES):
        prob = predictions[0][i] * 100
        st.progress(int(prob), text=f"{name}: {prob:.1f}%")
