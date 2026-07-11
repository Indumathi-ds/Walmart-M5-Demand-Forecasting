# Walmart-M5-Demand-Forecasting
End-to-end retail demand forecasting using the Walmart M5 dataset with LightGBM, feature engineering, and business insights.
# Walmart M5 Demand Forecasting using Machine Learning

##  Project Overview

Demand forecasting is a critical task in retail because accurate sales predictions help businesses optimize inventory, reduce stock shortages, minimize overstocking costs, and improve supply chain planning.

This project develops an end-to-end machine learning pipeline using the Walmart M5 Forecasting dataset to predict daily product sales. The project covers data preprocessing, feature engineering, model comparison, business analysis, and demand forecasting using LightGBM.

---

##  Business Problem

Retail businesses need accurate sales forecasts to:

- Reduce stock shortages
- Minimize excess inventory
- Improve supply chain planning
- Optimize staffing
- Improve customer satisfaction

This project predicts future daily sales using historical sales data, calendar information, pricing data, and engineered time-series features.

---

##  Dataset

**Dataset:** Walmart M5 Forecasting Dataset

Files used:

- calendar.csv
- sell_prices.csv
- sales_train_evaluation.csv

Dataset contains:

- Historical daily sales
- Product information
- Store information
- Calendar events
- SNAP program information
- Product prices

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- LightGBM
- Joblib

---

##  Project Workflow

### 1. Data Loading

Loaded three datasets:

- Sales
- Calendar
- Selling Prices

### 2. Data Preprocessing

- Converted wide sales data into long format using `pd.melt()`
- Merged sales, calendar, and price datasets
- Handled missing values
- Forward-filled missing prices
- Removed unnecessary columns

### 3. Feature Engineering

Created important time-series features:

- Lag 1
- Lag 7
- Rolling Mean (7 days)
- Rolling Standard Deviation (7 days)

### 4. Data Preparation

- Label Encoding
- Chronological Train-Test Split

Training Period:

**2011 – 2015**

Testing Period:

**2016**

### 5. Machine Learning Models

Baseline Model

- Random Forest Regressor

Final Model

- LightGBM Regressor

---

##  Model Performance

| Model | MAE | RMSE |
|--------|-----:|------:|
| Random Forest | 1.98 | 4.38 |
| LightGBM      | 0.94 | 2.05 |

LightGBM achieved the best performance and was selected as the final forecasting model.

---

##  Business Insights

### Product Category Analysis

- FOODS is the highest-selling category.
- Grocery products contribute the majority of total sales.

### Weekly Sales Analysis

- Saturday and Sunday recorded the highest sales.
- Retailers should increase inventory before weekends.

### Price Analysis

- Selling price has a weak negative correlation with sales (-0.134).
- Historical demand influences sales more than price changes.

### SNAP Analysis

Average Sales

| SNAP | Average Sales |
|------:|--------------:|
| No | 1.094 |
| Yes | 1.193 |

Sales increased by approximately **9%** on SNAP days.

------------------------------------------------------------------------------

##  Business Recommendations

- Increase inventory for food products.
- Prepare additional stock before weekends.
- Monitor SNAP periods for higher demand.
- Use historical demand patterns rather than price alone for forecasting.

---

##  Repository Structure

```
Walmart-M5-Demand-Forecasting/
│
├── M5_Demand_Forecasting.ipynb
├── README.md
├── requirements.txt
├── lightgbm_model.pkl
└── images/
```

---

##  Future Improvements

- Deploy using Streamlit
- Train on higher-memory infrastructure using the complete M5 dataset
- Experiment with XGBoost and CatBoost
- Automate daily demand forecasting

---

## Author

**Indumathi Balasubramaniyan**
M.Sc. Statistics

Aspiring Data Scientist
