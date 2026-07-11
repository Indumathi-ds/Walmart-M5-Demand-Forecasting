import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Walmart M5 Demand Forecasting",
    page_icon="🛒",
    layout="wide"
)

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
# Technologies
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
    st.metric("MAE", "0.94")

with col2:
    st.metric("RMSE", "2.05")

st.info("LightGBM achieved better performance than Random Forest on the Walmart M5 dataset.")

st.divider()

# -----------------------------
# Business Value
# -----------------------------
st.header("💼 Business Value")

st.write("""
This forecasting system can help retailers:

- Improve inventory planning
- Reduce stock-outs
- Reduce excess inventory
- Improve demand forecasting accuracy
- Support better business decisions
""")

st.divider()

# -----------------------------
# Business Insights
# -----------------------------
st.header("📊 Business Insights")

tab1, tab2, tab3 = st.tabs([
    "Category Sales",
    "Weekday Sales",
    "SNAP Analysis"
])

with tab1:

    st.subheader("Total Sales by Category")

    category_sales = {
        "FOODS": 3366443,
        "HOUSEHOLD": 1465505,
        "HOBBIES": 889141
    }

    st.bar_chart(category_sales)

    st.success("FOODS category contributes the highest sales.")


with tab2:

    st.subheader("Sales by Weekday")

    weekday_sales = {
        "Saturday":1036392,
        "Sunday":1031171,
        "Friday":818223,
        "Monday":779594,
        "Tuesday":694154,
        "Thursday":684728,
        "Wednesday":676827
    }

    st.bar_chart(weekday_sales)

    st.success("Weekend sales are higher than weekdays.")


with tab3:

    st.subheader("SNAP Impact")

    snap_sales = {
        "Non SNAP":1.09,
        "SNAP":1.19
    }

    st.bar_chart(snap_sales)

    st.success("Average sales are slightly higher on SNAP days.")
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
