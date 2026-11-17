
import pandas as pd
from pathlib import Path

# Step 1: Locate the processed dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

rfm = pd.read_csv(DATA_DIR / "customer_rfm.csv")

# Step 2: Calculate recency and monetary scores
# Higher recency score means a more recent purchase.
rfm["R_score"] = pd.qcut(
    rfm["recency"].rank(method="first"),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["M_score"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Frequency is heavily concentrated at 1 order.
# Use explicit thresholds instead of quantiles.
def frequency_score(frequency):
    if frequency >= 5:
        return 5
    elif frequency == 4:
        return 4
    elif frequency == 3:
        return 3
    elif frequency == 2:
        return 2
    else:
        return 1

rfm["F_score"] = rfm["frequency"].apply(
    frequency_score
)

# Step 3: Combine scores
rfm["RFM_score"] = (
    rfm["R_score"]
    + rfm["F_score"]
    + rfm["M_score"]
)

# Step 4: Assign business-friendly segments
def assign_segment(row):
    r = row["R_score"]
    f = row["F_score"]
    m = row["M_score"]

    if r >= 4 and f >= 2 and m >= 4:
        return "Champions"
    elif f >= 2 and r >= 3:
        return "Loyal Customers"
    elif r >= 4 and f == 1:
        return "Potential Loyalists"
    elif r <= 2 and f >= 2:
        return "At Risk"
    else:
        return "Others"

rfm["segment"] = rfm.apply(
    assign_segment,
    axis=1
)

# Step 5: Display results
print("CUSTOMER SEGMENTATION COMPLETED")
print("--------------------------------")

print("\nCustomers per segment:")
print(rfm["segment"].value_counts())

print("\nAverage RFM metrics per segment:")
print(
    rfm.groupby("segment")[
        ["recency", "frequency", "monetary"]
    ].mean().round(2)
)

# Step 6: Save results
output_path = DATA_DIR / "customer_segments.csv"
rfm.to_csv(output_path, index=False)

print("\nSaved to:", output_path)
