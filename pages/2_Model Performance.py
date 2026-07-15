import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance")

st.write("""
This page summarizes the performance of the machine learning models
used for Walmart M5 Demand Forecasting.
""")

st.divider()

# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

st.header("🏆 Model Comparison")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Random Forest")

    st.metric(
        "MAE",
        "1.98"
    )

    st.metric(
        "RMSE",
        "4.38"
    )

with col2:

    st.subheader("LightGBM")

    st.metric(
        "MAE",
        "1.65"
    )

    st.metric(
        "RMSE",
        "4.16"
    )

st.success(
    "LightGBM achieved lower MAE and RMSE than Random Forest, making it the final selected model."
)

st.divider()

# -------------------------------------------------
# WHY LIGHTGBM
# -------------------------------------------------

st.header("✅ Why LightGBM?")

st.write("""

LightGBM was selected because it:

- Produces lower prediction error
- Handles large datasets efficiently
- Trains faster than Random Forest
- Captures complex feature interactions
- Provides robust performance for demand forecasting

""")

st.divider()

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

st.header("📊 Top Important Features")

feature_importance = {
    "Rolling Mean (7 Days)":0.41,
    "Rolling Std (7 Days)":0.13,
    "Week Number":0.10,
    "Lag 1":0.07,
    "Lag 7":0.06,
    "Month":0.05,
    "Weekday":0.03,
    "Selling Price":0.03,
    "SNAP":0.01
}

st.bar_chart(feature_importance)

st.info(
    "Historical sales patterns contributed more to forecasting accuracy than price or SNAP information."
)

st.divider()

# -------------------------------------------------
# MODEL PIPELINE
# -------------------------------------------------

st.header("⚙️ Machine Learning Pipeline")

st.markdown("""

1. Data Cleaning

2. Missing Value Handling

3. Feature Engineering

- Lag Features
- Rolling Mean
- Rolling Standard Deviation

4. Label Encoding

5. Train-Test Split

6. Model Training

- Random Forest
- LightGBM

7. Model Evaluation

- MAE
- RMSE

""")

st.divider()

# -------------------------------------------------
# BUSINESS IMPACT
# -------------------------------------------------

st.header("💼 Business Impact")
st.divider()

st.write("""
This demand forecasting solution can help retailers:

- 📦 Improve inventory planning
- 📉 Reduce stock-outs
- 📈 Reduce excess inventory
- 🎯 Improve demand forecasting accuracy
- 💰 Reduce inventory holding costs
- 🛍 Support smarter replenishment decisions
- 📊 Enable data-driven business planning
""")

st.divider()

# -------------------------------------------------
# CONCLUSION
# -------------------------------------------------

st.header("📌 Conclusion")

st.success("""

LightGBM outperformed Random Forest for Walmart M5 demand forecasting.

The model effectively utilized historical sales patterns, calendar events,
pricing information and engineered lag features to improve forecasting
accuracy.

""")

st.divider()

st.caption(
    "Developed by Indumathi Balasubramaniyan | Walmart M5 Demand Forecasting"
)
