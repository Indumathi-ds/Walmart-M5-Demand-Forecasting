import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="🤖"
)

st.title("🤖 Walmart Sales Prediction")

# Load model and encoders
model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.write("Enter the product details below.")

# -----------------------------
# Inputs
# -----------------------------

item_id = st.text_input("Item ID")

dept_id = st.selectbox(
    "Department",
    ["FOODS_1", "FOODS_2", "FOODS_3",
     "HOBBIES_1", "HOBBIES_2",
     "HOUSEHOLD_1", "HOUSEHOLD_2"]
)

cat_id = st.selectbox(
    "Category",
    ["FOODS", "HOBBIES", "HOUSEHOLD"]
)

store_id = st.selectbox(
    "Store",
    ["CA_1"]
)

weekday = st.selectbox(
    "Weekday",
    ["Monday","Tuesday","Wednesday","Thursday",
     "Friday","Saturday","Sunday"]
)

sell_price = st.number_input(
    "Selling Price",
    min_value=0.0,
    value=5.0
)

lag_1 = st.number_input(
    "Yesterday Sales",
    min_value=0.0,
    value=1.0
)

lag_7 = st.number_input(
    "Last Week Sales",
    min_value=0.0,
    value=1.0
)

rolling_mean_7 = st.number_input(
    "7-Day Average Sales",
    min_value=0.0,
    value=1.0
)

rolling_std_7 = st.number_input(
    "7-Day Sales Std",
    min_value=0.0,
    value=0.5
)

predict = st.button("Predict Sales")

if predict:
    st.info("Prediction logic will be added in the next step.")
