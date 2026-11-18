
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load Olist datasets
customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

payments = pd.read_csv(
    RAW_DIR / "olist_order_payments_dataset.csv"
)

# Convert purchase dates
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Use delivered orders
orders = orders[
    orders["order_status"] == "delivered"
].copy()

# Calculate payment totals per order
order_payments = (
    payments.groupby("order_id", as_index=False)
    ["payment_value"].sum()
)

# Combine data
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

# Historical snapshot dates
cutoff_dates = [
    "2017-04-01",
    "2017-07-01",
    "2017-10-01",
    "2018-01-01",
    "2018-04-01"
]

all_snapshots = []

for date in cutoff_dates:

    cutoff = pd.Timestamp(date)
    prediction_end = cutoff + pd.Timedelta(days=90)

    # Historical transactions
    historical = df[
        df["order_purchase_timestamp"] < cutoff
    ].copy()

    # Future transactions for labels
    future = df[
        (df["order_purchase_timestamp"] >= cutoff)
        & (df["order_purchase_timestamp"] < prediction_end)
    ].copy()

    # Historical customer features
    features = historical.groupby(
        "customer_unique_id"
    ).agg(
        last_purchase=("order_purchase_timestamp", "max"),
        first_purchase=("order_purchase_timestamp", "min"),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum")
    ).reset_index()

    features["recency"] = (
        cutoff - features["last_purchase"]
    ).dt.days

    features["customer_tenure"] = (
        cutoff - features["first_purchase"]
    ).dt.days

    features["average_order_value"] = (
        features["monetary"] / features["frequency"]
    )

    # Create repeat-purchase labels
    future_customers = set(
        future["customer_unique_id"].unique()
    )

    features["repeat_purchase_90d"] = (
        features["customer_unique_id"]
        .isin(future_customers)
        .astype(int)
    )

    features["cutoff_date"] = cutoff

    features = features[
        [
            "customer_unique_id",
            "cutoff_date",
            "recency",
            "frequency",
            "monetary",
            "customer_tenure",
            "average_order_value",
            "repeat_purchase_90d"
        ]
    ]

    all_snapshots.append(features)

    print(
        f"{date}: {len(features)} customers, "
        f"{features['repeat_purchase_90d'].sum()} repeat purchasers"
    )

# Combine all historical snapshots
prediction_data = pd.concat(
    all_snapshots,
    ignore_index=True
)

output_path = (
    PROCESSED_DIR / "customer_time_based_data.csv"
)

prediction_data.to_csv(output_path, index=False)

print("\nTIME-BASED DATA PREPARATION COMPLETED")
print("Total snapshot records:", len(prediction_data))
print("Saved to:", output_path)
