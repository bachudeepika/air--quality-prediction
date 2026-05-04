import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model.joblib")

st.set_page_config(page_title="AQI Prediction", layout="centered")

st.title("🌍 Air Quality Index (AQI) Prediction")
st.write("Enter pollution values to predict AQI")

st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------

st.subheader("Pollutant Values")

pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
pm10 = st.number_input("PM10", min_value=0.0, value=80.0)
no2 = st.number_input("NO2", min_value=0.0, value=30.0)
so2 = st.number_input("SO2", min_value=0.0, value=20.0)
co = st.number_input("CO", min_value=0.0, value=1.0)
o3 = st.number_input("O3", min_value=0.0, value=40.0)

st.subheader("Date Information")

date_input = st.date_input("Select Date")

st.subheader("City")

# 🔹 Replace with your actual encoding mapping
city_mapping = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Hyderabad": 4
}

city = st.selectbox("Select City", list(city_mapping.keys()))
city_encoded = city_mapping[city]

st.subheader("Previous AQI Statistics")

aqi_7day_avg = st.number_input("7 Day AQI Average", min_value=0.0, value=100.0)
aqi_7day_std = st.number_input("7 Day AQI Std Dev", min_value=0.0, value=15.0)

st.markdown("---")

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict AQI"):

    # Date Features
    year = date_input.year
    month = date_input.month
    day = date_input.day
    dayofweek = date_input.weekday()
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if dayofweek >= 5 else 0

    # Engineered Features
    pm25_pm10_ratio = pm25 / pm10 if pm10 != 0 else 0
    no2_so2_ratio = no2 / so2 if so2 != 0 else 0
    total_pm = pm25 + pm10
    pollution_score = pm25 + pm10 + no2 + so2 + co + o3

    # Feature Order MUST MATCH TRAINING
    feature_cols = [
        'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
        'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend',
        'city_encoded', 'pm25_pm10_ratio', 'no2_so2_ratio',
        'total_pm', 'pollution_score',
        'aqi_7day_avg', 'aqi_7day_std'
    ]

    input_data = pd.DataFrame([[
        pm25, pm10, no2, so2, co, o3,
        year, month, day, dayofweek, quarter, is_weekend,
        city_encoded, pm25_pm10_ratio, no2_so2_ratio,
        total_pm, pollution_score,
        aqi_7day_avg, aqi_7day_std
    ]], columns=feature_cols)

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted AQI: {prediction:.2f}")

    # AQI Category
    if prediction <= 100:
        category = "Good 😊"
    elif prediction <= 150:
        category = "Satisfactory 🙂"
    elif prediction <= 200:
        category = "Moderate 😷"
    elif prediction <= 300:
        category = "Poor ⚠"
    elif prediction <= 400:
        category = "Very Poor 🚨"
    else:
        category = "Severe 🔴"

    st.subheader(f"AQI Category: {category}")