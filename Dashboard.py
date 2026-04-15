import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import json
import os

# ------------------ CONFIG ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.title("🐟 Fish Species Classification App")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model_safe():
    model_path = r"/Users/imuppala/Desktop/Guvi_Greeshma/Best_Model.h5"   # best ResNet50 model

    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    return load_model(model_path)

model = load_model_safe()

# ------------------ Class Names (must match training labels) ------------------
class_names = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 
    'fish sea_food sea_bass', 'fish sea_food shrimp', 
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

# ------------------ IMAGE PREPROCESSING ------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: matches ResNet50 training
    img_array = preprocess_input(img_array)

    return img_array

# ------------------ UPLOAD IMAGE ------------------
uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    with st.spinner("Predicting..."):
        predictions = model.predict(processed_image)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions)

    st.success(f"✅ Prediction: {predicted_class}")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

    # ------------------ TOP 3 ------------------
    st.subheader("Top 3 Predictions")

    top3_idx = predictions[0].argsort()[-3:][::-1]

    for i in top3_idx:
        st.write(f"{class_names[i]} : {predictions[0][i]*100:.2f}%")

    # ------------------ PROBABILITY CHART ------------------
    st.subheader("Class Probabilities")
    st.bar_chart(predictions[0])

