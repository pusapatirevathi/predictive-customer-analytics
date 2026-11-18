
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "repeat_purchase_model.joblib"
)

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)


class PredictionAgent:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run "
                "python analytics/save_prediction_model.py"
            )

        artifact = joblib.load(MODEL_PATH)

        self.model = artifact["model"]
        self.features = artifact["features"]

    def predict(self, customer_data):
        """
        Generate repeat-purchase ranking scores.
        """

        missing = [
            col for col in self.features
            if col not in customer_data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing customer features: {missing}"
            )

        results = customer_data.copy()

        results["repeat_purchase_score"] = (
            self.model.predict_proba(
                results[self.features]
            )[:, 1]
        )

        return results


if __name__ == "__main__":

    print("Starting Prediction Agent...")

    agent = PredictionAgent()

    df = pd.read_csv(DATA_PATH)

    # Use the final historical snapshot
    # as a demonstration dataset.
    customers = df[
        df["cutoff_date"] == "2018-04-01"
    ].copy()

    predictions = agent.predict(customers)

    print("\nPREDICTION AGENT RESULTS")
    print(
        predictions[
            [
                "customer_unique_id",
                "repeat_purchase_score"
            ]
        ].head(10)
    )

    print("\nPREDICTION AGENT COMPLETED")
