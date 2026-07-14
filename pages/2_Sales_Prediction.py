import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Sales Prediction", page_icon="🤖")

st.title("🤖 Walmart Sales Prediction")

st.write("Predict daily demand using the trained LightGBM model.")

# Load model and encoders
model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.divider()

st.subheader("Product Information")

st.text_input("Store", value="CA_1", disabled=True)
st.text_input("Category", value="HOBBIES", disabled=True)
st.text_input("Department", value="HOBBIES_1", disabled=True)

item_id = st.text_input("Item ID", "HOBBIES_1_001")

date = st.date_input(
    "Forecast Date",
    datetime.date.today()
)

sell_price = st.number_input(
    "Selling Price",
    min_value=0.0,
    value=5.0
)

lag_1 = st.number_input(
    "Yesterday Sales",
    min_value=0.0,
    value=2.0
)

lag_7 = st.number_input(
    "Sales 7 Days Ago",
    min_value=0.0,
    value=2.0
)

rolling_mean_7 = st.number_input(
    "7-Day Average Sales",
    min_value=0.0,
    value=2.0
)

rolling_std_7 = st.number_input(
    "7-Day Sales Std",
    min_value=0.0,
    value=0.5
)

if st.button("Predict Sales"):

    weekday = date.strftime("%A")
    month = date.month
    year = date.year
    wday = date.weekday() + 1

    row = pd.DataFrame({
        "item_id":[item_id],
        "dept_id":["HOBBIES_1"],
        "cat_id":["HOBBIES"],
        "store_id":["CA_1"],
        "state_id":["CA"],
        "wm_yr_wk":[1],
        "weekday":[weekday],
        "wday":[wday],
        "month":[month],
        "year":[year],
        "event_name_1":["No Event"],
        "event_type_1":["No Event"],
        "event_name_2":["No Event"],
        "event_type_2":["No Event"],
        "snap_CA":[0],
        "snap_TX":[0],
        "snap_WI":[0],
        "sell_price":[sell_price],
        "lag_1":[lag_1],
        "lag_7":[lag_7],
        "rolling_mean_7":[rolling_mean_7],
        "rolling_std_7":[rolling_std_7]
    })

    categorical = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"
    ]

    try:

        for col in categorical:
            row[col] = encoders[col].transform(row[col])

        prediction = model.predict(row)[0]

        st.success(f"Predicted Sales: {prediction:.2f} units")

    except Exception as e:
        st.error(
            "Unknown Item ID. Please use an Item ID that existed in the training data."
        )
