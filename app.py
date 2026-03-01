import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="🦠",
    layout="centered"
)

# ---------------- CUSTOM UI DESIGN ----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: red;
    }
    .result-box {
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:22px;
        font-weight:bold;
        margin-top:20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>
    🦠 Malaria Cell Detection App
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Upload a blood cell image to detect infection instantly.
    </p>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malaria_model.keras", compile=False)

model = load_model()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Blood Cell Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Center Image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img, use_column_width=True)

    # Preprocess
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction with spinner
    with st.spinner("🔍 Analyzing Image..."):
        time.sleep(1)
        prediction = model.predict(img_array)[0][0]

    # Class Mapping
    # Parasitized = 0
    # Uninfected = 1

    if prediction < 0.5:
        label = "❌ Cell is Infected (Parasitized)"
        confidence = 1 - prediction
        color = "#ff4b4b"
    else:
        label = "✅ Cell is Healthy (Uninfected)"
        confidence = prediction
        color = "#00FFAA"

    # Stylish Result Box
    st.markdown(f"""
        <div class="result-box" style="background-color:#111; border:2px solid {color}; color:{color};">
            {label} <br><br>
            Confidence: {confidence*100:.2f}%
        </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Made with ❤️ by Gajender Kumar")
