
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

# Step 1: Load historical snapshots
df = pd.read_csv(DATA_PATH)

df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

# Step 2: Define training and testing periods
train_dates = pd.to_datetime([
    "2017-04-01",
    "2017-07-01",
    "2017-10-01"
])

test_date = pd.Timestamp("2018-04-01")

train_df = df[df["cutoff_date"].isin(train_dates)].copy()
test_df = df[df["cutoff_date"] == test_date].copy()

# Step 3: Select features
features = [
    "recency",
    "frequency",
    "monetary",
    "customer_tenure",
    "average_order_value"
]

X_train = train_df[features]
y_train = train_df["repeat_purchase_90d"]

X_test = test_df[features]
y_test = test_df["repeat_purchase_90d"]

print("TIME-BASED MODEL VALIDATION")
print("---------------------------")
print("Training snapshots:", len(train_df))
print("Testing snapshots:", len(test_df))
print("Training positives:", y_train.sum())
print("Testing positives:", y_test.sum())

# Step 4: Check target classes
if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both training and testing data must contain "
        "positive and negative examples."
    )

# Step 5: Train the model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluate
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nROC-AUC:")
print(round(roc_auc_score(y_test, probabilities), 4))

print("\nAVERAGE PRECISION:")
print(round(average_precision_score(y_test, probabilities), 4))

print("\nBASELINE POSITIVE RATE:")
print(round(y_test.mean(), 4))

print("\nTIME-BASED VALIDATION COMPLETED")
