
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# =====================================
# STEP 1: Define project paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# STEP 2: Load customer snapshots
# =====================================

df = pd.read_csv(DATA_PATH)

df["cutoff_date"] = pd.to_datetime(
    df["cutoff_date"]
)

# =====================================
# STEP 3: Create chronological split
# =====================================

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

# =====================================
# STEP 4: Select input features
# =====================================

feature_columns = [
    "recency",
    "frequency",
    "customer_tenure"
]

target_column = "repeat_purchase_90d"

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

if (
    X_train.empty
    or X_test.empty
    or y_train.nunique() != 2
    or y_test.nunique() != 2
):
    raise ValueError(
        "Training and testing datasets must "
        "be nonempty and contain both classes."
    )

print("MODEL COMPARISON")
print("----------------")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Training positives:", y_train.sum())
print("Testing positives:", y_test.sum())

# =====================================
# STEP 5: Define machine learning models
# =====================================

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

# =====================================
# STEP 6: Train and evaluate models
# =====================================

results = []

for model_name, model in models.items():

    print("\n==============================")
    print("Training:", model_name)
    print("==============================")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    avg_precision = average_precision_score(
        y_test,
        probabilities
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("ROC-AUC:", round(roc_auc, 4))
    print(
        "Average Precision:",
        round(avg_precision, 4)
    )

    results.append({
        "model": model_name,
        "roc_auc": roc_auc,
        "average_precision": avg_precision
    })

# =====================================
# STEP 7: Compare model performance
# =====================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="average_precision",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print("----------------------")
print(results_df.to_string(index=False))

print("\nBaseline positive rate:")
print(round(y_test.mean(), 4))

# =====================================
# STEP 8: Save comparison results
# =====================================

output_path = (
    REPORTS_DIR / "model_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nMODEL COMPARISON COMPLETED")
print("Saved to:", output_path)
