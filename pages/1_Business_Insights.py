import streamlit as st

st.set_page_config(page_title="Business Insights", page_icon="📊")

st.title("📊 Business Insights")

st.write("Business insights generated from the Walmart M5 dataset.")

st.divider()

# -----------------------------
# Category Sales
# -----------------------------
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

    st.success("FOODS category contributes the highest total sales.")


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

    st.subheader("Average Sales on SNAP Days")

    snap_sales = {
        "Non SNAP":1.09,
        "SNAP":1.19
    }

    st.bar_chart(snap_sales)

    st.success("Average sales are slightly higher on SNAP days.")
