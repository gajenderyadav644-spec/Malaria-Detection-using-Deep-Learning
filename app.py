import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page settings
st.set_page_config(page_title="Malaria Cell Detection", page_icon="🦠")

st.title("🦠 Malaria Cell Detection App")
st.write("Upload a blood cell image to check whether it is infected or not.")

# Load model safely
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "malaria_model.keras",
        compile=False
    )
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image (must match training size)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.write("Raw Prediction:", float(prediction))  # Debug line

    # IMPORTANT: Class mapping
    # Parasitized = 0
    # Uninfected = 1

    if prediction < 0.5:
        label = "❌ Infected (Parasitized)"
        confidence = 1 - prediction
        st.error(label)
    else:
        label = "✅ Uninfected"
        confidence = prediction
        st.success(label)

    st.write(f"Confidence: {confidence:.2f}")
