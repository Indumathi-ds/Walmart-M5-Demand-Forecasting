import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Walmart Sales Prediction")

st.write("""
Predict daily sales using the trained **LightGBM** model.

This model was trained on Walmart M5 historical sales data.
""")

st.divider()

# ----------------------------------------
# Load Model & Encoders
# ----------------------------------------
model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")
item_ids = joblib.load("item_ids.pkl")

# ----------------------------------------
# Product Information
# ----------------------------------------

st.header("Product Information")

col1, col2 = st.columns(2)

with col1:

    item_id = st.selectbox(
        "Item ID",
        item_ids
    )

    dept_id = "HOBBIES_1"

    cat_id = "HOBBIES"

    store_id = "CA_1"

    state_id = "CA"

with col2:

    forecast_date = st.date_input(
        "Forecast Date",
        value=date.today()
    )

    sell_price = st.number_input(
        "Selling Price",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

st.divider()

# ----------------------------------------
# Historical Sales Features
# ----------------------------------------

st.header("Historical Sales")

col1, col2 = st.columns(2)

with col1:

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

with col2:

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

st.divider()

predict = st.button("Predict Sales")

if predict:

    weekday = forecast_date.strftime("%A")

    month = forecast_date.month

    year = forecast_date.year

    wday = forecast_date.weekday() + 1

    wm_yr_wk = 1

    event_name_1 = "No Event"
    event_type_1 = "No Event"

    event_name_2 = "No Event"
    event_type_2 = "No Event"

    snap_CA = 0
    snap_TX = 0
    snap_WI = 0

    input_df = pd.DataFrame({

        "item_id":[item_id],
        "dept_id":[dept_id],
        "cat_id":[cat_id],
        "store_id":[store_id],
        "state_id":[state_id],
        "wm_yr_wk":[wm_yr_wk],
        "weekday":[weekday],
        "wday":[wday],
        "month":[month],
        "year":[year],
        "event_name_1":[event_name_1],
        "event_type_1":[event_type_1],
        "event_name_2":[event_name_2],
        "event_type_2":[event_type_2],
        "snap_CA":[snap_CA],
        "snap_TX":[snap_TX],
        "snap_WI":[snap_WI],
        "sell_price":[sell_price],
        "lag_1":[lag_1],
        "lag_7":[lag_7],
        "rolling_mean_7":[rolling_mean_7],
        "rolling_std_7":[rolling_std_7]

    })

    categorical_cols = [

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



        # ----------------------------------------

        # Encode categorical variables

        # ----------------------------------------



        for col in categorical_cols:

            input_df[col] = encoders[col].transform(input_df[col])



        # ----------------------------------------

        # Reorder columns exactly as training data

        # ----------------------------------------



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



        # ----------------------------------------

        # Prediction

        # ----------------------------------------



        prediction = model.predict(input_df)[0]



        st.divider()



        st.success("Prediction Completed Successfully")



        st.metric(

            label="📦 Predicted Sales",

            value=f"{prediction:.2f} Units"

        )



        # ----------------------------------------

        # Business Recommendation

        # ----------------------------------------



        if prediction >= 10:



            st.warning("""

### Recommendation



High demand is expected.



- Increase inventory.

- Monitor stock levels closely.

- Consider replenishment before demand peaks.

""")



        elif prediction >= 5:



            st.info("""

### Recommendation



Moderate demand is expected.



- Maintain normal inventory levels.

- Continue regular monitoring.

""")



        else:



            st.success("""

### Recommendation



Low demand is expected.



- Avoid overstocking.

- Reduce unnecessary inventory holding costs.

""")



        st.divider()



        st.subheader("Prediction Summary")



        summary = pd.DataFrame({

            "Feature": [

                "Item ID",

                "Department",

                "Category",

                "Store",

                "Forecast Date",

                "Selling Price",

                "Yesterday Sales",

                "Last Week Sales",

                "7-Day Average",

                "7-Day Std"

            ],

            "Value": [

                item_id,

                dept_id,

                cat_id,

                store_id,

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



        st.error(

            "Prediction failed. Please ensure the selected Item ID exists in the training data."

        )



        st.exception(e)



st.divider()



st.caption(

    "Developed by Indumathi Balasubramaniyan | Walmart M5 Demand Forecasting"

)
