import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from src.model_training import train_digit_model
from src.image_pipeline import preprocess_image

# 1. Page Configuration for Responsiveness
st.set_page_config(
    layout="wide", 
    page_title="Digitix Dashboard",
    initial_sidebar_state="expanded"
)

# 2. Enhanced Responsive CSS
st.markdown("""
    <style>
    /* Force columns to stack on screens smaller than 768px */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 80% !important;
            flex: 1 1 100% !important;
            min-width: 80% !important;
            margin-bottom: 20px;
        }
    }
    
    /* Dashboard Box Styling */
    [data-testid="column"] {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }

    /* SPECIFIC BOX SIZE: 150px Width x 200px Height */
    .img-container img {        
        width: 150px !important;
        height: 200px !important;
        object-fit: contain; /* Maintains aspect ratio within the box */
        border-radius: 4px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Center text in the predicted digit display */
    .prediction-text {
        text-align: Left;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        color: #1f77b4;
    }
    .st-emotion-cache-1n6tfoc {
        display: flex;
        gap: 1rem;
        width: 80%;
        max-width: 80%;
        height: auto;
        min-width: 1rem;
        flex-flow: column;
        flex: 1 1 0%;
        -webkit-box-align: start;
        align-items: start;
        -webkit-box-pack: start;
        justify-content: start;
        overflow: visible;
    }
    .st-emotion-cache-3uj0rx h3 {
        font-size: 0.90rem;
        font-weight: 600;
        padding: 0.75rem 0px 1rem;
    }
    .st-emotion-cache-1vo6xi6 {
        width: 100%;
        height: fit-content;
        max-width: 100%;
        min-width: 1rem;
        position: relative;
        overflow: visible;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: SYSTEM & HYPERPARAMS ---
with st.sidebar:
    st.header("⚙️ System Control")
    model_path = os.path.join('models', 'digit_model.h5')
    
    if os.path.exists(model_path):
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ No Model Found")

    st.divider()
    st.subheader("Hyperparameters")
    opt = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    loss_fn = st.selectbox("Loss Function", ["sparse_categorical_crossentropy", "categorical_crossentropy"])
    met = st.selectbox("Metrics", ["accuracy", "mse"])
    ep = st.slider("Epochs", 1, 20, 5)
    
    if st.button("Retrain Model", use_container_width=True):
        with st.status("Training CNN..."):
            train_digit_model(optimizer=opt, loss=loss_fn, metrics=[met], epochs=ep)
        st.session_state['trained'] = True
        st.rerun()

    if os.path.exists('models/accuracy_plot.png'):
        st.image('models/accuracy_plot.png', use_container_width=True)


# --- MAIN DASHBOARD GRID ---
main_container = st.container()

with main_container:
    file = st.file_uploader("📤 Image Upload", type=["png", "jpg", "jpeg"])
    
    # THE 4-GRID PANEL
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("📸 Input Image")
        if file:
            img = Image.open(file)
            # Apply the 100x120 container here
            resized_image = img.resize((100, 120))
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(resized_image)
            st.markdown('</div>', unsafe_allow_html=True)

    with row1_col2:
        st.subheader("🖼️ Pre-processed Output")
        if file:
            processed = preprocess_image(img)
            # Apply the 100x120 container here as well
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            image_data = (np.squeeze(processed[0]) * 255).astype(np.uint8)
            resized_image = Image.fromarray(image_data).resize((100, 120))
            st.image(resized_image)
            st.markdown('</div>', unsafe_allow_html=True)

   
    # Second Row: Results & Stats
    st.divider()
    row2_col1, row2_col2 = st.columns([1, 1], gap="medium")

    with row2_col1:
        st.subheader("🔢 Predicted Digit")
        if file and os.path.exists(model_path):
            @st.cache_resource
            def load_model_cached(path):
                return tf.keras.models.load_model(path)
            
            model = load_model_cached(model_path)
            preds = model.predict(processed)
            digit = np.argmax(preds)
            st.markdown(f"<h1 class='prediction-text' style='font-size: 6vw;'>{digit}</h1>", unsafe_allow_html=True)
        else:
            st.write("Result will appear here.")

    with row2_col2:
        st.subheader("📊 Classification Probability")
        if file:
            chart_data = {str(i): float(preds[0][i]) for i in range(10)}  
            st.bar_chart(chart_data, use_container_width=True, height=250)