
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(
    DATA_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    DATA_DIR / "olist_orders_dataset.csv"
)

# Combine orders with customer identifiers
customer_orders = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one"
)

# Keep delivered orders for initial purchase analysis
customer_orders = customer_orders[
    customer_orders["order_status"] == "delivered"
].copy()

# Convert purchase dates to datetime
customer_orders["order_purchase_timestamp"] = (
    pd.to_datetime(
        customer_orders["order_purchase_timestamp"]
    )
)

# Create a basic customer profile
profiles = customer_orders.groupby(
    "customer_unique_id"
).agg(
    total_orders=("order_id", "nunique"),
    first_purchase=("order_purchase_timestamp", "min"),
    last_purchase=("order_purchase_timestamp", "max")
).reset_index()

print(profiles.head())

print("\nNumber of customer profiles:")
print(len(profiles))

profiles.to_csv(
    OUTPUT_DIR / "customer_profiles.csv",
    index=False
)

print("\nCustomer profiles saved successfully!")
