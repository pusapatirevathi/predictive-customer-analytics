
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# STEP 1: File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_time_based_data.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load data
df = pd.read_csv(DATA_PATH)
df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

# STEP 3: Chronological training and validation
train_dates = pd.to_datetime([
    "2017-04-01",
    "2017-07-01",
    "2017-10-01"
])

validation_date = pd.Timestamp("2018-01-01")

train_df = df[df["cutoff_date"].isin(train_dates)].copy()
val_df = df[df["cutoff_date"] == validation_date].copy()

# April 2018 remains reserved for final testing.

# STEP 4: Seven customer features
features = [
    "recency",
    "frequency",
    "customer_tenure",
    "days_between_first_last",
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d"
]

target = "repeat_purchase_90d"

X_train = train_df[features]
y_train = train_df[target]

X_val = val_df[features]
y_val = val_df[target]

if (
    X_train.empty
    or X_val.empty
    or y_train.nunique() != 2
    or y_val.nunique() != 2
):
    raise ValueError("Invalid training or validation data.")

# STEP 5: Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

# STEP 6: Train and evaluate
results = []

print("\nIMPROVED MODEL COMPARISON")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Validation positive rate:", round(y_val.mean(), 4))

for name, model in models.items():

    print("\n==============================")
    print("Model:", name)
    print("==============================")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = model.predict(X_val)

    roc_auc = roc_auc_score(y_val, probabilities)
    avg_precision = average_precision_score(
        y_val, probabilities
    )

    print("\nClassification Report:")
    print(classification_report(
        y_val, predictions, zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("ROC-AUC:", round(roc_auc, 4))
    print("Average Precision:", round(avg_precision, 4))

    results.append({
        "model": name,
        "roc_auc": roc_auc,
        "average_precision": avg_precision
    })

# STEP 7: Save comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="average_precision",
    ascending=False
)

print("\nFINAL VALIDATION COMPARISON")
print(results_df.to_string(index=False))

output_path = REPORTS_DIR / "improved_model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nIMPROVED MODEL COMPARISON COMPLETED")
print("Saved to:", output_path)
