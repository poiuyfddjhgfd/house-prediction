import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("my_california_housing_model.pkl")

st.set_page_config(page_title="California Housing Price Prediction")

st.title("🏠 California Housing Price Prediction App")

st.write("Enter housing details to predict house price")

# User inputs

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)

# ocean proximity (same encoding used during training)
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
)

# Convert ocean proximity to number (example encoding)
ocean_map = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "NEAR OCEAN": 2,
    "NEAR BAY": 3,
    "ISLAND": 4
}

ocean_proximity_val = ocean_map[ocean_proximity]


# Prediction button

if st.button("Predict Price"):

    features = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity_val
    ]])

    prediction = model.predict(features)

    st.success(f"🏡 Predicted House Price: ${prediction[0]:,.2f}")
