# 🌱 Crop Recommendation System

A Machine Learning-based Crop Recommendation System that predicts the most suitable crop based on soil nutrients and environmental conditions.

## Features
- User-friendly Streamlit interface
- Machine Learning prediction
- Compares multiple Scikit-learn models
- Automatically selects the best-performing model

## Dataset
The model is trained using the Crop Recommendation Dataset.

**Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall
**Target:** Crop Label

## 🛠️ Tech Stack & Libraries
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn, NumPy, Pandas
- **Data Visualization:** Seaborn, Matplotlib
- **Model Serialization:** Pickle

## Project Structure

Crop-Recommendation-System/
│
├── app.py
├── train_model.py
├── eda.py
├── requirements.txt
├── data/
├── models/
└── output/

## Installation
```bash
git clone https://github.com/sindhuturkio/Crop-Recommendation-System.git
cd Crop-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

## Machine Learning Models Evaluated
- Decision Tree — 98.64%
- Random Forest — 99.32% ✅ (Selected)
- K-Nearest Neighbors (KNN) — 97.05%

## Author
Sindhu Turkio
