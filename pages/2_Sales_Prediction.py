import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Walmart Sales Prediction")

st.markdown("""
Predict daily sales using the trained **LightGBM** model.
""")

st.divider()

# ----------------------------------
# Load Model
# ----------------------------------

model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")
item_ids = joblib.load("item_ids.pkl")

# ----------------------------------
# Product Information
# ----------------------------------

st.header("Product Information")

col1, col2 = st.columns(2)

with col1:

    item_id = st.selectbox(
        "Item ID",
        item_ids
    )

    st.text_input(
        "Department",
        value="HOBBIES_1",
        disabled=True
    )

    st.text_input(
        "Category",
        value="HOBBIES",
        disabled=True
    )

with col2:

    st.text_input(
        "Store",
        value="CA_1",
        disabled=True
    )

    forecast_date = st.date_input(
        "Forecast Date",
        value=date.today()
    )

st.divider()

# ----------------------------------
# Sales Features
# ----------------------------------

st.header("Sales Features")

col1, col2 = st.columns(2)

with col1:

    sell_price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    lag_1 = st.number_input(
        "Yesterday Sales",
        min_value=0.0,
        value=2.0
    )

with col2:

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
    "7-Day Sales Standard Deviation",
    min_value=0.0,
    value=0.5
)

predict = st.button("Predict Sales")

if predict:

    weekday = forecast_date.strftime("%A")

    month = forecast_date.month

    year = forecast_date.year

    wday = forecast_date.weekday() + 1

    input_df = pd.DataFrame({

        "item_id":[item_id],
        "dept_id":["HOBBIES_1"],
        "cat_id":["HOBBIES"],
        "store_id":["CA_1"],
        "state_id":["CA"],
        "wm_yr_wk":[11101],
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

        # -------------------------------
        # Encode categorical columns
        # -------------------------------

        for col in categorical:
            input_df[col] = encoders[col].transform(input_df[col])

        # -------------------------------
        # Reorder columns
        # -------------------------------

        input_df = input_df[[
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "sell_price",
            "lag_1",
            "lag_7",
            "rolling_mean_7",
            "rolling_std_7"
        ]]

        # -------------------------------
        # Predict
        # -------------------------------

        prediction = model.predict(input_df)[0]

        st.divider()

        st.success("Prediction Completed Successfully")

        st.metric(
            label="📦 Predicted Sales",
            value=f"{prediction:.2f} Units"
        )

        # -------------------------------
        # Business Recommendation
        # -------------------------------

        if prediction >= 10:

            st.warning("""
### 📈 High Demand

Increase inventory.

Monitor stock regularly.

Prepare replenishment.
""")

        elif prediction >= 5:

            st.info("""
### 📊 Moderate Demand

Maintain normal inventory levels.
""")

        else:

            st.success("""
### 📉 Low Demand

Avoid overstocking.

Maintain minimum inventory.
""")

        st.divider()

        st.subheader("Prediction Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Item ID",
                "Forecast Date",
                "Selling Price",
                "Yesterday Sales",
                "Sales 7 Days Ago",
                "7-Day Average",
                "7-Day Std"
            ],

            "Value":[
                item_id,
                forecast_date,
                sell_price,
                lag_1,
                lag_7,
                rolling_mean_7,
                rolling_std_7
            ]

        })

        st.dataframe(summary, use_container_width=True)

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)

st.divider()

st.caption(
    "Developed by Indumathi Balasubramaniyan | Walmart M5 Demand Forecasting"
)
