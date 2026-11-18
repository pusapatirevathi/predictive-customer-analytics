
import pandas as pd
from pathlib import Path

# ==========================================
# STEP 1: Define project folders
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 2: Load Olist datasets
# ==========================================

customers = pd.read_csv(
    RAW_DIR / "olist_customers_dataset.csv"
)

orders = pd.read_csv(
    RAW_DIR / "olist_orders_dataset.csv"
)

payments = pd.read_csv(
    RAW_DIR / "olist_order_payments_dataset.csv"
)


# ==========================================
# STEP 3: Prepare order dates
# ==========================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

# Do not filter using final order status.
# The goal is to predict future recorded orders,
# not necessarily successful deliveries.

orders = orders.dropna(
    subset=["order_purchase_timestamp"]
).copy()


# ==========================================
# STEP 4: Calculate payment totals
# ==========================================

order_payments = (
    payments.groupby(
        "order_id",
        as_index=False
    )["payment_value"].sum()
)


# ==========================================
# STEP 5: Combine customer and order data
# ==========================================

df = orders.merge(
    customers[
        ["customer_id", "customer_unique_id"]
    ],
    on="customer_id",
    how="inner"
)

# Use a left join so orders without payment
# records are not excluded from the analysis.

df = df.merge(
    order_payments,
    on="order_id",
    how="left"
)

# Missing payment totals are treated as zero.
# This is a modeling assumption, not proof
# that the order had no payment.

df["payment_value"] = (
    df["payment_value"].fillna(0)
)


# ==========================================
# STEP 6: Define historical cutoff dates
# ==========================================

cutoff_dates = [
    "2017-04-01",
    "2017-07-01",
    "2017-10-01",
    "2018-01-01",
    "2018-04-01"
]

prediction_days = 90

all_snapshots = []


# ==========================================
# STEP 7: Generate historical snapshots
# ==========================================

for date in cutoff_dates:

    cutoff = pd.Timestamp(date)

    prediction_end = (
        cutoff + pd.Timedelta(days=prediction_days)
    )

    # Historical orders before cutoff
    historical = df[
        df["order_purchase_timestamp"] < cutoff
    ].copy()

    # Future orders used only for labels
    future = df[
        (df["order_purchase_timestamp"] >= cutoff)
        &
        (df["order_purchase_timestamp"] < prediction_end)
    ].copy()

    if historical.empty:
        print(f"Skipping {date}: no historical orders")
        continue

    # ======================================
    # STEP 8: Calculate customer features
    # ======================================

    features = historical.groupby(
        "customer_unique_id"
    ).agg(
        last_purchase=(
            "order_purchase_timestamp",
            "max"
        ),
        first_purchase=(
            "order_purchase_timestamp",
            "min"
        ),
        frequency=(
            "order_id",
            "nunique"
        ),
        monetary=(
            "payment_value",
            "sum"
        )
    ).reset_index()

    # Days since last recorded order
    features["recency"] = (
        cutoff - features["last_purchase"]
    ).dt.days

    # Days since first recorded order
    features["customer_tenure"] = (
        cutoff - features["first_purchase"]
    ).dt.days

    # Average payment value per order
    features["average_order_value"] = (
        features["monetary"] / features["frequency"]
    )

    # ======================================
    # STEP 9: Create prediction labels
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
    # STEP 10: Select final columns
    # ======================================

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
    ].copy()

    all_snapshots.append(features)

    repeat_customers = (
        features["repeat_purchase_90d"].sum()
    )

    print(
        f"{date}: {len(features)} customers, "
        f"{repeat_customers} repeat purchasers"
    )


# ==========================================
# STEP 11: Combine historical snapshots
# ==========================================

if not all_snapshots:
    raise ValueError(
        "No historical snapshots were generated."
    )

prediction_data = pd.concat(
    all_snapshots,
    ignore_index=True
)


# ==========================================
# STEP 12: Save prediction dataset
# ==========================================

output_path = (
    PROCESSED_DIR / "customer_time_based_data.csv"
)

prediction_data.to_csv(
    output_path,
    index=False
)


# ==========================================
# STEP 13: Display results
# ==========================================

print("\nTIME-BASED DATA PREPARATION COMPLETED")
print("-------------------------------------")

print("Total snapshot records:", len(prediction_data))

print("\nPrediction label counts:")
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
