
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load dataset
df = pd.read_csv(DATA_PATH)
df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

# STEP 3: Split data chronologically
# Training labels must have matured before
# the April 2018 test cutoff.

train_dates = pd.to_datetime([
    "2017-04-01",
    "2017-07-01",
    "2017-10-01"
])

test_date = pd.Timestamp("2018-04-01")

train_df = df[
    df["cutoff_date"].isin(train_dates)
].copy()

test_df = df[
    df["cutoff_date"] == test_date
].copy()

# STEP 4: Select features
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

X_test = test_df[features]
y_test = test_df[target]

if (
    X_train.empty
    or X_test.empty
    or y_train.nunique() != 2
    or y_test.nunique() != 2
):
    raise ValueError("Invalid training or testing data.")

# STEP 5: Train selected model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# STEP 6: Generate predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

# STEP 7: Evaluate
roc_auc = roc_auc_score(y_test, probabilities)
avg_precision = average_precision_score(
    y_test, probabilities
)

print("\nFINAL TEST RESULTS")
print("------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Actual repeat purchasers:", int(y_test.sum()))
print("Baseline positive rate:", round(y_test.mean(), 4))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(avg_precision, 4))

# STEP 8: Save evaluation results
results = pd.DataFrame([{
    "model": "Logistic Regression",
    "training_records": len(X_train),
    "testing_records": len(X_test),
    "roc_auc": roc_auc,
    "average_precision": avg_precision,
    "baseline_positive_rate": y_test.mean()
}])

output_path = REPORTS_DIR / "final_model_results.csv"
results.to_csv(output_path, index=False)

print("\nFINAL MODEL TESTING COMPLETED")
print("Saved to:", output_path)
