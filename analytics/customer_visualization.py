
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load segmented customer data
df = pd.read_csv(DATA_DIR / "customer_segments.csv")

# Graph 1: Customers per segment
segment_counts = df["segment"].value_counts()

plt.figure(figsize=(10, 5))
segment_counts.plot(kind="bar")

plt.title("Customer Distribution by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "customer_segments.png",
    dpi=300
)
plt.close()

# Graph 2: Average spending by segment
average_spending = (
    df.groupby("segment")["monetary"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
average_spending.plot(kind="bar")

plt.title("Average Spending by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Payment Value (BRL)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "segment_spending.png",
    dpi=300
)
plt.close()

# Graph 3: Average recency by segment
average_recency = (
    df.groupby("segment")["recency"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 5))
average_recency.plot(kind="bar")

plt.title("Average Recency by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Days Since Last Purchase")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "segment_recency.png",
    dpi=300
)
plt.close()

print("CUSTOMER VISUALIZATIONS COMPLETED")
print("Saved 3 graphs to:", REPORT_DIR)
