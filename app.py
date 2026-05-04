import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Import our custom preprocessing logic
from src.inference import preprocess_image

# Page Config
st.set_page_config(page_title="Digitix - Digit Recognizer", page_icon="🔢")

# --- UI Header ---
st.title("🔢 Digitix: Handwritten Digit Recognizer")
st.markdown("""
Upload a handwritten digit image (0-9). The system will use a **Convolutional Neural Network (CNN)** to identify the digit.
""")

# --- Sidebar: Model Status ---
st.sidebar.header("System Status")
model_path = os.path.join('models', 'digit_model.h5')

if os.path.exists(model_path):
    st.sidebar.success("✅ Model Loaded Successfully")
    # Load model once and cache it for performance
    @st.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(model_path)
    
    model = load_my_model()
else:
    st.sidebar.error("❌ Model File Not Found")
    st.sidebar.info("Please run 'python src/model_training.py' first.")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Original Image
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)

    # 2. Preprocess & Predict
    with st.spinner('Analyzing...'):
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        
        result = np.argmax(predictions)
        confidence = np.max(predictions)

    # 3. Display Results
    with col2:
        st.metric(label="Predicted Digit", value=str(result))
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Show mini version of what the model actually "sees" (the 28x28 version)
        st.write("What the CNN sees:")
        st.image(processed_img[0], width=70)

    # 4. Confidence Chart
    st.divider()
    st.subheader("Classification Probability")
    chart_data = {str(i): float(predictions[0][i]) for i in range(10)}
    st.bar_chart(chart_data)