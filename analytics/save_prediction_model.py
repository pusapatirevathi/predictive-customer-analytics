
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# STEP 1: Define paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load dataset
df = pd.read_csv(DATA_PATH)
df["cutoff_date"] = pd.to_datetime(df["cutoff_date"])

# STEP 3: Use historical training snapshots
# These labels mature before April 2018.
train_dates = pd.to_datetime([
    "2017-04-01",
    "2017-07-01",
    "2017-10-01"
])

train_df = df[
    df["cutoff_date"].isin(train_dates)
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

if X_train.empty or y_train.nunique() != 2:
    raise ValueError("Invalid training dataset.")

# STEP 5: Build model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# STEP 6: Train
print("Training final Logistic Regression model...")
model.fit(X_train, y_train)

# STEP 7: Save model and feature names
model_path = MODEL_DIR / "repeat_purchase_model.joblib"

artifact = {
    "model": model,
    "features": features,
    "target": target,
    "training_cutoffs": [
        date.strftime("%Y-%m-%d")
        for date in train_dates
    ]
}

joblib.dump(artifact, model_path)

print("\nMODEL SAVED SUCCESSFULLY")
print("Saved to:", model_path)
print("Features:", features)

# STEP 8: Verify saved model
loaded_artifact = joblib.load(model_path)

sample = X_train.head(5)

probabilities = loaded_artifact["model"].predict_proba(
    sample[loaded_artifact["features"]]
)[:, 1]

print("\nSample repeat-purchase scores:")
print(probabilities)

print("\nMODEL VERIFICATION COMPLETED")
