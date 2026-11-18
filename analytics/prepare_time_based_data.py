
import pandas as pd
from pathlib import Path

# ==========================================
# STEP 1: Define folders
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# STEP 2: Load data
# ==========================================

customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

# ==========================================
# STEP 3: Prepare order timestamps
# ==========================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

orders = orders.dropna(
    subset=["order_purchase_timestamp"]
).copy()

# Do not use final delivery or cancellation
# status to select historical orders.

df = orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="inner"
)

# ==========================================
# STEP 4: Define snapshot dates
# ==========================================

cutoff_dates = [
    "2017-04-01",
    "2017-07-01",
    "2017-10-01",
    "2018-01-01",
    "2018-04-01"
]

all_snapshots = []

# ==========================================
# STEP 5: Generate customer snapshots
# ==========================================

for date in cutoff_dates:

    cutoff = pd.Timestamp(date)
    prediction_end = cutoff + pd.Timedelta(days=90)

    historical = df[
        df["order_purchase_timestamp"] < cutoff
    ].copy()

    future = df[
        (df["order_purchase_timestamp"] >= cutoff)
        & (df["order_purchase_timestamp"] < prediction_end)
    ].copy()

    if historical.empty:
        continue

    # Basic customer features
    features = historical.groupby(
        "customer_unique_id"
    ).agg(
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique")
    ).reset_index()

    features["recency"] = (
        cutoff - features["last_purchase"]
    ).dt.days

    features["customer_tenure"] = (
        cutoff - features["first_purchase"]
    ).dt.days

    features["days_between_first_last"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days

    # ======================================
    # STEP 6: Recent order activity
    # ======================================

    for days in [30, 60, 90]:

        window_start = cutoff - pd.Timedelta(days=days)

        recent_orders = historical[
            historical["order_purchase_timestamp"] >= window_start
        ]

        counts = (
            recent_orders.groupby("customer_unique_id")["order_id"]
            .nunique()
        )

        column_name = f"orders_last_{days}d"

        features[column_name] = (
            features["customer_unique_id"]
            .map(counts)
            .fillna(0)
            .astype(int)
        )

    # ======================================
    # STEP 7: Create target labels
    # ======================================

    future_customers = set(
        future["customer_unique_id"].unique()
    )

    features["repeat_purchase_90d"] = (
        features["customer_unique_id"]
        .isin(future_customers)
        .astype(int)
    )

    features["cutoff_date"] = cutoff

    # ======================================
    # STEP 8: Select final columns
    # ======================================

    features = features[
        [
            "customer_unique_id",
            "cutoff_date",
            "recency",
            "frequency",
            "customer_tenure",
            "days_between_first_last",
            "orders_last_30d",
            "orders_last_60d",
            "orders_last_90d",
            "repeat_purchase_90d"
        ]
    ]

    all_snapshots.append(features)

    print(
        f"{date}: {len(features)} customers, "
        f"{features['repeat_purchase_90d'].sum()} repeat purchasers"
    )

# ==========================================
# STEP 9: Save final dataset
# ==========================================

if not all_snapshots:
    raise ValueError("No snapshots generated.")

prediction_data = pd.concat(
    all_snapshots,
    ignore_index=True
)

output_path = (
    PROCESSED_DIR / "customer_time_based_data.csv"
)

prediction_data.to_csv(output_path, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("Total snapshot records:", len(prediction_data))
print("Columns:", prediction_data.columns.tolist())
print("Saved to:", output_path)
