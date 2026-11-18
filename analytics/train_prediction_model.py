
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Step 1: Locate project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Step 2: Load prediction dataset
df = pd.read_csv(
    DATA_DIR / "customer_prediction_data.csv"
)

# Step 3: Select input features
feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "customer_tenure",
    "average_order_value"
]

X = df[feature_columns]
y = df["repeat_purchase_90d"]

print("MODEL TRAINING")
print("--------------")
print("Total customers:", len(df))
print("\nTarget distribution:")
print(y.value_counts())

# Step 4: Verify both classes exist
if y.nunique() < 2:
    raise ValueError(
        "The target contains only one class. "
        "Check your prediction window and dataset."
    )

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 6: Build the machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 9: Evaluate the model
print("\nCLASSIFICATION REPORT")
print("---------------------")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC SCORE")
print(round(roc_auc_score(y_test, y_prob), 4))

print("\nAVERAGE PRECISION")
print(round(
    average_precision_score(y_test, y_prob),
    4
))

print("\nBASELINE POSITIVE RATE")
print(round(y_test.mean(), 4))

print("\nMODEL TRAINING COMPLETED")
