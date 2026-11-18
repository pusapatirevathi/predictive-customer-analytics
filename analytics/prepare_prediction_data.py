
import pandas as pd
from pathlib import Path

# Step 1: Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Step 2: Load the Olist datasets
customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

payments = pd.read_csv(
    RAW_DIR / "olist_order_payments_dataset.csv"
)

# Step 3: Convert purchase dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Step 4: Keep delivered orders
orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Step 5: Calculate payment totals per order
order_payments = (
    payments.groupby("order_id", as_index=False)
    ["payment_value"].sum()
)

# Step 6: Combine orders, customers, and payments
df = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="inner"
)

df = df.merge(
    order_payments,
    on="order_id",
    how="inner"
)

# Step 7: Define historical cutoff and future window
cutoff_date = pd.Timestamp("2018-04-01")
prediction_end = cutoff_date + pd.Timedelta(days=90)

# Historical data only
historical = df[
    df["order_purchase_timestamp"] < cutoff_date
].copy()

# Future purchases used only to create labels
future = df[
    (df["order_purchase_timestamp"] >= cutoff_date)
    & (df["order_purchase_timestamp"] < prediction_end)
].copy()

# Step 8: Create historical customer features
features = historical.groupby(
    "customer_unique_id"
).agg(
    last_purchase=("order_purchase_timestamp", "max"),
    first_purchase=("order_purchase_timestamp", "min"),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
).reset_index()

features["recency"] = (
    cutoff_date - features["last_purchase"]
).dt.days

features["customer_tenure"] = (
    cutoff_date - features["first_purchase"]
).dt.days

features["average_order_value"] = (
    features["monetary"] / features["frequency"]
)

# Step 9: Create prediction labels
future_customers = set(
    future["customer_unique_id"].unique()
)

features["repeat_purchase_90d"] = (
    features["customer_unique_id"]
    .isin(future_customers)
    .astype(int)
)

# Step 10: Select final columns
prediction_data = features[
    [
        "customer_unique_id",
        "recency",
        "frequency",
        "monetary",
        "customer_tenure",
        "average_order_value",
        "repeat_purchase_90d"
    ]
]

# Step 11: Save prediction dataset
output_path = (
    PROCESSED_DIR / "customer_prediction_data.csv"
)

prediction_data.to_csv(output_path, index=False)

# Step 12: Display results
print("PREDICTION DATA PREPARATION COMPLETED")
print("-------------------------------------")

print("Historical cutoff:", cutoff_date.date())
print("Prediction window ends:", prediction_end.date())
print("Customers:", len(prediction_data))

print("\nPrediction labels:")
print(
    prediction_data["repeat_purchase_90d"]
    .value_counts()
)

print("\nPrediction label percentages:")
print(
    prediction_data["repeat_purchase_90d"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFirst five records:")
print(prediction_data.head())

print("\nSaved to:", output_path)
