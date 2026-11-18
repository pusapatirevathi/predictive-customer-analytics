
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# Step 1: Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "customer_prediction_data.csv"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

# Step 2: Load customer prediction dataset
df = pd.read_csv(DATA_PATH)

features = [
    "recency",
    "frequency",
    "monetary",
    "customer_tenure",
    "average_order_value"
]

target = "repeat_purchase_90d"

X = df[features]
y = df[target]

if y.nunique() != 2:
    raise ValueError("Prediction target must contain two classes.")

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 4: Train the baseline model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Step 5: Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nMODEL EVALUATION COMPLETED")
print("--------------------------")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, probabilities), 4
))

print("Average Precision:", round(
    average_precision_score(y_test, probabilities), 4
))

print("Baseline Positive Rate:", round(
    y_test.mean(), 4
))

# Step 6: Generate confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["No Repeat", "Repeat"],
    cmap="Blues"
)

plt.title("Repeat Purchase Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Step 7: Generate ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Repeat Purchase Prediction - ROC Curve")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
plt.close()

# Step 8: Generate precision-recall curve
PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.axhline(
    y=y_test.mean(),
    linestyle="--",
    label="Baseline"
)

plt.title("Repeat Purchase - Precision Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    REPORTS_DIR / "precision_recall_curve.png",
    dpi=150
)
plt.close()

print("\nGraphs saved successfully to reports/")
