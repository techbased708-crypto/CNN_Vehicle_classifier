import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# ── Model Load ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cars_cnn.keras")
    return model

model = load_model()

# ── Class Names ──────────────────────────────────────────
# Kaggle dataset: mmohaiminulislam/vehicles-image-dataset
# Alphabetical order (same as TensorFlow reads folders)
CLASS_NAMES = ['Bus', 'Car', 'Motorcycle', 'Truck', 'Van']

st.title("🚗 Vehicle Image Classifier")
st.markdown("CNN model se koi bhi vehicle image upload karo aur result dekho!")
st.markdown("---")

uploaded_file = st.file_uploader("Image upload karo", type=["jpg", "png", "jpeg", "jfif"])

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
        num_classes = predictions.shape[1]
        CLASS_NAMES_ADJUSTED = CLASS_NAMES[:num_classes]
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = CLASS_NAMES_ADJUSTED[predicted_index] if predicted_index < len(CLASS_NAMES_ADJUSTED) else f"Class {predicted_index}"
        confidence = np.max(predictions[0]) * 100     

    # All class probabilities
    st.markdown("### 📊 All Predictions:")
    for i, name in enumerate(CLASS_NAMES):
        prob = predictions[0][i] * 100
        st.progress(int(prob), text=f"{name}: {prob:.1f}%")
