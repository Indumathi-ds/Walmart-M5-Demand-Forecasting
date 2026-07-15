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

@st.cache_data
def load_sales():
    df = pd.read_csv("prediction_lookup.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

sales = load_sales()

model = joblib.load("lightgbm_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.header("Select Product")

item_list = sorted(sales["item_id"].unique())
selected_item = st.selectbox("Item ID", item_list)

item_df = sales[sales["item_id"] == selected_item].copy()

available_dates = sorted(item_df["date"].dt.strftime("%Y-%m-%d").unique())
selected_date = st.selectbox("Historical Date", available_dates)

selected_row = item_df[item_df["date"].dt.strftime("%Y-%m-%d") == selected_date].iloc[0]

st.divider()
predict = st.button("Predict Sales", type="primary")

if predict:
    input_df = pd.DataFrame({
        "item_id":[selected_row["item_id"]],
        "dept_id":[selected_row["dept_id"]],
        "cat_id":[selected_row["cat_id"]],
        "store_id":[selected_row["store_id"]],
        "state_id":[selected_row["state_id"]],
        "wm_yr_wk":[selected_row["wm_yr_wk"]],
        "weekday":[selected_row["weekday"]],
        "wday":[selected_row["wday"]],
        "month":[selected_row["month"]],
        "year":[selected_row["year"]],
        "event_name_1":[selected_row["event_name_1"]],
        "event_type_1":[selected_row["event_type_1"]],
        "event_name_2":[selected_row["event_name_2"]],
        "event_type_2":[selected_row["event_type_2"]],
        "snap_CA":[selected_row["snap_CA"]],
        "snap_TX":[selected_row["snap_TX"]],
        "snap_WI":[selected_row["snap_WI"]],
        "sell_price":[selected_row["sell_price"]],
        "lag_1":[selected_row["lag_1"]],
        "lag_7":[selected_row["lag_7"]],
        "rolling_mean_7":[selected_row["rolling_mean_7"]],
        "rolling_std_7":[selected_row["rolling_std_7"]],
    })

    categorical_cols = [
        "item_id","dept_id","cat_id","store_id","state_id",
        "weekday","event_name_1","event_type_1","event_name_2","event_type_2"
    ]

    try:
        for col in categorical_cols:
            input_df[col] = encoders[col].transform(input_df[col])

        feature_order = [
            "item_id","dept_id","cat_id","store_id","state_id","wm_yr_wk",
            "weekday","wday","month","year","event_name_1","event_type_1",
            "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI",
            "sell_price","lag_1","lag_7","rolling_mean_7","rolling_std_7"
        ]

        input_df = input_df[feature_order]
        prediction = model.predict(input_df)[0]

        st.divider()
        st.success("Prediction Completed Successfully ✅")
        st.metric(label="📦 Predicted Sales", value=f"{prediction:.2f} Units")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.sidebar.title("📈 Walmart M5")
