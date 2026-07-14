import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Walmart M5 Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛒 Walmart M5")

st.sidebar.info(
    """
    **Retail Demand Forecasting**

    Developed by:

    **Indumathi Balasubramaniyan**
    """
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("lightgbm_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🛒 Walmart M5 Demand Forecasting")

st.markdown("""
### Retail Demand Forecasting using Machine Learning

This project predicts daily product demand for Walmart stores using the **LightGBM** machine learning algorithm.

The objective is to help retailers forecast future demand accurately, reduce stock shortages, minimize overstocking, and improve inventory planning.
""")

st.divider()

# -----------------------------
# Dataset
# -----------------------------
st.header("📂 Dataset")

st.write("""
- **Dataset:** Walmart M5 Forecasting Dataset
- **Source:** Kaggle
- Daily sales data of Walmart products
- Calendar information
- Product prices
- Event and SNAP information
""")

st.divider()

# -----------------------------
# Technologies Used
# -----------------------------
st.header("🛠 Technologies Used")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
- Python
- Pandas
- NumPy
- Scikit-learn
""")

with col2:
    st.markdown("""
- LightGBM
- Streamlit
- Matplotlib
- Joblib
""")

st.divider()

# -----------------------------
# Model Performance
# -----------------------------
st.header("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error (MAE)", "0.94")

with col2:
    st.metric("Root Mean Squared Error (RMSE)", "2.05")

st.success("✅ LightGBM achieved better performance than Random Forest on the Walmart M5 Demand Forecasting dataset.")

st.divider()

# -----------------------------
# Business Value
# -----------------------------
st.header("💼 Business Value")

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

# -----------------------------
# Navigation
# -----------------------------
st.header("🚀 Explore the Project")

st.info("""
Use the navigation menu on the left to explore:

📊 **Business Insights**
- Category-wise Sales
- Weekday-wise Sales
- SNAP Analysis

🤖 **Sales Prediction**
- Predict product demand using the trained LightGBM model.
""")
