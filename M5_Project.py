#  Walmart M5 Demand Forecasting using Machine Learning
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the Dataset
calendar=pd.read_csv(r"C:\Users\Hema Kalai\Downloads\M5 dataset\calendar.csv")
sales=pd.read_csv(r"C:\Users\Hema Kalai\Downloads\M5 dataset\sales_train_evaluation.csv")
price=pd.read_csv(r"C:\Users\Hema Kalai\Downloads\M5 dataset\sell_prices.csv")

#Data preprocessing

sales=pd.melt(
    sales,id_vars=sales.columns[:6],var_name="days",value_name="sales"
)
sales.shape

sales_calendar = pd.merge(
    sales,calendar,left_on="days",right_on="d",how="left"
)

sales=pd.merge(sales_calendar,price,on=["store_id","item_id","wm_yr_wk"],how="left")

sales["sell_price"].isna().sum()
sales[sales["sell_price"].isna()].head()
sales.isnull().sum()
sales=sales.sort_values(by=["item_id","store_id","date"])
sales["sell_price"]=(sales.groupby(["item_id","store_id"])["sell_price"].ffill().bfill())
sales["lag_1"]=(sales.groupby(["item_id","store_id"])["sales"].shift(1))
sales["lag_7"]=(sales.groupby(["item_id","store_id"])["sales"].shift(7))
sales["rolling_mean_7"]=(
    sales.groupby(["item_id","store_id"])["sales"]
                               .transform (lambda x:x.shift(1).rolling(7).mean()))
sales["rolling_std_7"]=(
    sales.groupby(["item_id","store_id"])["sales"]
                               .transform (lambda x:x.shift(1).rolling(7).std()))
sales[
 ["item_id","store_id","date","sales","lag_1","lag_7","rolling_mean_7","rolling_std_7"]].head(-10)


sales["event_name_1"]=sales["event_name_1"].fillna("No Event")
sales["event_type_1"]=sales["event_type_1"].fillna("No Event")
sales["event_name_2"]=sales["event_name_2"].fillna("No Event")
sales["event_type_2"]=sales["event_type_2"].fillna("No Event")
sales=sales.dropna()
sales=sales.drop(columns=["id","days","d"])

# Feature Engineering
x=sales.drop(columns=["sales"])
y=sales["sales"]

# train_test_split
print(sales["date"].min())
print(sales["date"].max())
sales["date"] = pd.to_datetime(sales["date"])

train = sales[sales["date"] < "2016-01-01"]
test = sales[sales["date"] >= "2016-01-01"]

X_train = train.drop(columns=["sales", "date"])
y_train = train["sales"]

X_test = test.drop(columns=["sales", "date"])
y_test = test["sales"]

#label encoding
from sklearn.preprocessing import LabelEncoder

categorical_cols = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name_1", "event_type_1",
    "event_name_2", "event_type_2"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()

    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

    encoders[col] = le

#Random forest

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
print(y_test.mean())

from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# hyper tunning

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best MAE:", -grid_search.best_score_)
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

from sklearn.metrics import mean_absolute_error, root_mean_squared_error
mae = mean_absolute_error(y_test, y_pred_best)
rmse = root_mean_squared_error(y_test, y_pred_best)
print("Tuned MAE:", mae)
print("Tuned RMSE:", rmse)

#!pip install lightgbm

# lightGBM
from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = root_mean_squared_error(y_test, y_pred_lgb)

print("LightGBM MAE:", mae_lgb)
print("LightGBM RMSE:", rmse_lgb)

#----------------------------Business insights---------------------------------
category_sales = (
    sales.groupby("cat_id")["sales"]
    .sum()
    .sort_values(ascending=False)
)
print(category_sales)
#plot
category_sales.plot(
    kind="bar",
    figsize=(6,4)
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()

weekday_sales = (
    sales.groupby("weekday")["sales"]
    .sum()
    .sort_values(ascending=False)
)
print(weekday_sales)

#plot
weekday_sales.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Total Sales by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Total Sales")
plt.show()

print(sales["state_id"].unique())
print(sales["store_id"].unique())
print(sales["cat_id"].unique())

sales[["sell_price", "sales"]].corr()

snap_sales = (
    sales.groupby("snap_CA")["sales"]
    .mean()
)
print(snap_sales)

#plot
snap_sales.plot(
    kind="bar",
    figsize=(5,4)
)

plt.title("Average Sales on SNAP vs Non-SNAP Days")
plt.xlabel("SNAP Day")
plt.ylabel("Average Sales")
plt.show()


# Conclusion
# Successfully developed an end-to-end demand forecasting pipeline using the Walmart M5 dataset.
# Engineered lag and rolling statistical features to improve forecasting performance.
# Compared Random Forest and LightGBM models.
# LightGBM achieved the best performance with MAE = 0.94 and RMSE = 2.05.
# Business insights showed:
       #  FOODS is the highest-selling category.
       #  Weekend sales are significantly higher.
       #  Price has only a weak negative relationship with sales.
       #  SNAP days increase average sales.
