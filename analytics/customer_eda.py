
import pandas as pd
from pathlib import Path

# Step 1: Locate the dataset folder
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Step 2: Load customer and order datasets
customers = pd.read_csv(
    DATA_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    DATA_DIR / "olist_orders_dataset.csv"
)

# Step 3: Convert purchase timestamps to dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Step 4: Merge customers and orders
customer_orders = orders.merge(
    customers,
    on="customer_id",
    how="inner"
)

# Step 5: Calculate basic metrics
total_customers = customer_orders[
    "customer_unique_id"
].nunique()

total_orders = customer_orders[
    "order_id"
].nunique()

print("CUSTOMER PURCHASE ANALYSIS")
print("--------------------------")
print("Unique customers:", total_customers)
print("Total orders:", total_orders)

# Step 6: Calculate monthly order volume
customer_orders["order_month"] = (
    customer_orders["order_purchase_timestamp"]
    .dt.to_period("M")
)

monthly_orders = (
    customer_orders.groupby("order_month")["order_id"]
    .nunique()
    .sort_index()
)

print("\nMONTHLY ORDER VOLUME")
print(monthly_orders.to_string())
