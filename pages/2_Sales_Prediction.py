import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Walmart M5 Sales Prediction")

st.write("""
Predict Walmart product demand using the trained LightGBM model.
""")

st.divider()

# -------------------------------------------------
# LOAD FILES
# -------------------------------------------------

@st.cache_data
def load_sales():
    df = pd.read_csv("processed_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

sales = load_sales()

model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# -------------------------------------------------
# PRODUCT SELECTION
# -------------------------------------------------

st.header("Select Product")

item_list = sorted(sales["item_id"].unique())

selected_item = st.selectbox(
    "Item ID",
    item_list
)

item_df = sales[
    sales["item_id"] == selected_item
].copy()

available_dates = sorted(
    item_df["date"].dt.strftime("%Y-%m-%d").unique()
)

selected_date = st.selectbox(
    "Historical Date",
    available_dates
)

selected_row = item_df[
    item_df["date"].dt.strftime("%Y-%m-%d")
    == selected_date
].iloc[0]

st.divider()

st.subheader("Selected Record")

display_cols = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "sell_price",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7"
]

st.dataframe(
    pd.DataFrame(selected_row[display_cols]).T,
    use_container_width=True
)

predict = st.button(
    "Predict Sales",
    type="primary"
)
if predict:
    
    input_df = pd.DataFrame({
        "item_id": [selected_row["item_id"]],
        "dept_id": [selected_row["dept_id"]],
        "cat_id": [selected_row["cat_id"]],
        "store_id": [selected_row["store_id"]],
        "state_id": [selected_row["state_id"]],
        "wm_yr_wk": [selected_row["wm_yr_wk"]],
        "weekday": [selected_row["weekday"]],
        "wday": [selected_row["wday"]],
        "month": [selected_row["month"]],
        "year": [selected_row["year"]],
        "event_name_1": [selected_row["event_name_1"]],
        "event_type_1": [selected_row["event_type_1"]],
        "event_name_2": [selected_row["event_name_2"]],
        "event_type_2": [selected_row["event_type_2"]],
        "snap_CA": [selected_row["snap_CA"]],
        "snap_TX": [selected_row["snap_TX"]],
        "snap_WI": [selected_row["snap_WI"]],
        "sell_price": [selected_row["sell_price"]],
        "lag_1": [selected_row["lag_1"]],
        "lag_7": [selected_row["lag_7"]],
        "rolling_mean_7": [selected_row["rolling_mean_7"]],
        "rolling_std_7": [selected_row["rolling_std_7"]]
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
    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])    
        
        feature_order = [
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
        ]
        

        input_df = input_df[feature_order]
        prediction = model.predict(input_df)[0]
        st.divider()

        st.success("Prediction Completed Successfully ✅")

        st.metric(
            label="📦 Predicted Sales",
            value=f"{prediction:.2f} Units"
        )

        st.divider()

        st.subheader("Business Recommendation")

        if prediction >= 10:

            st.warning("""
### High Demand Expected

• Increase inventory.

• Monitor stock availability closely.

• Replenish stock before demand increases.
""")

        elif prediction >= 5:

            st.info("""
### Moderate Demand Expected

• Maintain current inventory.

• Monitor daily sales.

• No urgent replenishment required.
""")

        else:

            st.success("""
### Low Demand Expected

• Avoid overstocking.

• Reduce holding costs.

• Maintain minimum stock level.
""")

        st.divider()

        st.subheader("Prediction Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Item ID",
                "Department",
                "Category",
                "Store",
                "State",
                "Historical Date",
                "Selling Price",
                "Yesterday Sales",
                "Last Week Sales",
                "7-Day Average",
                "7-Day Std",
                "Predicted Sales"
            ],

            "Value":[
                selected_row["item_id"],
                selected_row["dept_id"],
                selected_row["cat_id"],
                selected_row["store_id"],
                selected_row["state_id"],
                selected_date,
                round(selected_row["sell_price"],2),
                round(selected_row["lag_1"],2),
                round(selected_row["lag_7"],2),
                round(selected_row["rolling_mean_7"],2),
                round(selected_row["rolling_std_7"],2),
                round(prediction,2)
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )
    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("📈 Walmart M5")

st.sidebar.markdown("""
### Project Information

**Model**
- LightGBM Regressor

**Dataset**
- Walmart M5 Forecasting

**Forecast Features**
- Historical Sales
- Calendar Events
- SNAP
- Selling Price

---

Developed by

**Indumathi Balasubramaniyan**
""")

st.divider()

st.caption(
    "Walmart M5 Demand Forecasting | LightGBM | Streamlit"
)
