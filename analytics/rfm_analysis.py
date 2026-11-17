
import pandas as pd
from pathlib import Path

# Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

payments = pd.read_csv(
    RAW_DIR / "olist_order_payments_dataset.csv"
)

# Convert dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Use completed, delivered orders for RFM
orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Sum payment values per order
order_payments = (
    payments.groupby("order_id", as_index=False)
    ["payment_value"].sum()
)

# Merge orders with customer identifiers
customer_orders = orders.merge(
    customers[
        ["customer_id", "customer_unique_id"]
    ],
    on="customer_id",
    how="inner"
)

# Merge with payment totals
customer_orders = customer_orders.merge(
    order_payments,
    on="order_id",
    how="inner"
)

# Set analysis date relative to dataset
analysis_date = (
    customer_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

# Calculate RFM metrics
rfm = customer_orders.groupby(
    "customer_unique_id"
).agg(
    last_purchase=("order_purchase_timestamp", "max"),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
).reset_index()

rfm["recency"] = (
    analysis_date - rfm["last_purchase"]
).dt.days

# Keep only RFM fields
rfm = rfm[
    ["customer_unique_id", "recency", "frequency", "monetary"]
]

# Save results
output_path = OUTPUT_DIR / "customer_rfm.csv"
rfm.to_csv(output_path, index=False)

print("RFM ANALYSIS COMPLETED")
print("----------------------")
print("Customers analyzed:", len(rfm))
print("\nFirst five records:")
print(rfm.head())

print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nSaved to:", output_path)
