import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# ----------------------------
# Load Your Trained Model
# ----------------------------
# Yeh line aapke 'models' folder se saved Random Forest model ko load karegi
@st.cache_resource
def load_model():
    with open("models/crop_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "❌ Model not found.\n\n"
        "Please run `train_model.py` first to train the model and generate `models/crop_model.pkl`."
    )
    st.stop()

# ----------------------------
# Header Section
# ----------------------------
st.title("🌱  Crop Recommendation System")
st.write("Enter the soil and environmental metrics below to find the most suitable crop for cultivation.")
st.divider()

# ----------------------------
# User Inputs UI
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧪 Soil Metrics")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50, step=1)
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    st.markdown("### 🌤️ Climate Metrics")
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

st.divider()

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("Recommend Best Crop 🌾", use_container_width=True):
    # Inputs ko model ke format (2D Array) mein convert karna
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Apne trained model se predict karwana
    prediction = model.predict(input_features)
    recommended_crop = prediction[0]
    
    # Result display karna
    st.balloons()
    st.success(f"### 🎉 Recommended Crop: **{recommended_crop.upper()}**")
    st.info("This recommendation is based on the trained Random Forest Machine Learning model.")