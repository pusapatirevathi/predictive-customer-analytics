
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load RFM customer data
df = pd.read_csv(DATA_DIR / "customer_rfm.csv")

# Select features for clustering
features = df[["recency", "frequency", "monetary"]].copy()

# Log transformation reduces the impact of outliers
features = np.log1p(features)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Test different numbers of clusters
inertias = []
silhouette_scores = []
k_values = range(2, 9)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(scaled_features)
    inertias.append(model.inertia_)

    # Sample for faster silhouette evaluation
    score = silhouette_score(
        scaled_features,
        labels,
        sample_size=5000,
        random_state=42
    )
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "kmeans_silhouette.png",
    dpi=300
)
plt.close()

# Select K with the highest sampled silhouette score
best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

# Train final clustering model
final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = final_model.fit_predict(
    scaled_features
)

# Save clustered customers
output_path = DATA_DIR / "customer_clusters.csv"
df.to_csv(output_path, index=False)

# Display cluster summary
print("K-MEANS CLUSTERING COMPLETED")
print("----------------------------")
print("Selected number of clusters:", best_k)

print("\nCustomers per cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nAverage RFM values per cluster:")
print(
    df.groupby("cluster")[
        ["recency", "frequency", "monetary"]
    ].mean().round(2)
)

print("\nSaved to:", output_path)
