
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

files = {
    "Customers": "olist_customers_dataset.csv",
    "Orders": "olist_orders_dataset.csv",
    "Order Items": "olist_order_items_dataset.csv",
    "Payments": "olist_order_payments_dataset.csv",
    "Reviews": "olist_order_reviews_dataset.csv",
    "Products": "olist_products_dataset.csv"
}

for name, filename in files.items():

    print(f"\n{'=' * 40}")
    print(f"Dataset: {name}")
    print(f"{'=' * 40}")

    df = pd.read_csv(DATA_DIR / filename)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())
