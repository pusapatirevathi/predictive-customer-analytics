
import pandas as pd
from pathlib import Path

from prediction_agent import PredictionAgent
from customer_profiling_agent import CustomerProfilingAgent

# =====================================
# STEP 1: Define project paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

# =====================================
# STEP 2: Recommendation Agent
# =====================================

class RecommendationAgent:

    def __init__(self):
        self.prediction_agent = PredictionAgent()
        self.profiling_agent = CustomerProfilingAgent()

        print("Recommendation Agent initialized.")

    def generate_recommendation(self, row):

        category = row["activity_category"]

        if category == "Recently Active":
            return "Personalized product recommendations"

        elif category == "Moderately Active":
            return "Customer engagement campaign"

        elif category == "At Risk":
            return "Re-engagement offer"

        else:
            return "Win-back campaign"

    def recommend(self, customer_data):

        # Generate customer profiles
        profiles = (
            self.profiling_agent.profile_customers(
                customer_data
            )
        )

        # Generate ML prediction scores
        predictions = (
            self.prediction_agent.predict(
                profiles
            )
        )

        # Generate marketing recommendations
        predictions["recommended_action"] = (
            predictions.apply(
                self.generate_recommendation,
                axis=1
            )
        )

        # Rank customers by repeat-purchase score
        predictions = predictions.sort_values(
            by="repeat_purchase_score",
            ascending=False
        )

        return predictions


# =====================================
# STEP 3: Test Recommendation Agent
# =====================================

if __name__ == "__main__":

    print("\nStarting Recommendation Agent...")

    agent = RecommendationAgent()

    df = pd.read_csv(DATA_PATH)

    customers = df[
        df["cutoff_date"] == "2018-04-01"
    ].copy()

    recommendations = agent.recommend(customers)

    print("\nRECOMMENDATION AGENT RESULTS")

    print(
        recommendations[
            [
                "customer_unique_id",
                "activity_category",
                "repeat_purchase_score",
                "recommended_action"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nRECOMMENDATION DISTRIBUTION")

    print(
        recommendations[
            "recommended_action"
        ].value_counts()
    )

    print("\nRECOMMENDATION AGENT COMPLETED")
