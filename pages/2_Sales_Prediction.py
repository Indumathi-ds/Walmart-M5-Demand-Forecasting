import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="🤖"
)

st.title("🤖 Walmart Sales Prediction")

st.write("Enter the product details below to predict sales.")

# Load trained model
model = joblib.load("lightgbm_model.pkl")

st.divider()

# --------------------------
# User Inputs
# --------------------------

store = st.selectbox(
    "Store",
    ["CA_1"]
)

category = st.selectbox(
    "Category",
    ["FOODS","HOUSEHOLD","HOBBIES"]
)

price = st.number_input(
    "Selling Price",
    min_value=0.0,
    value=5.0
)

weekday = st.selectbox(
    "Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

month = st.slider(
    "Month",
    1,
    12,
    6
)

snap = st.selectbox(
    "SNAP Day",
    ["No","Yes"]
)

st.button("Predict Sales")
