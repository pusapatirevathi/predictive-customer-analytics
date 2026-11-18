
import pandas as pd
from pathlib import Path

from recommendation_agent import RecommendationAgent


# ==========================================
# STEP 1: Define project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# STEP 2: Create Orchestrator Agent
# ==========================================

class OrchestratorAgent:

    def __init__(self):

        print("Initializing Orchestrator Agent...")

        self.recommendation_agent = RecommendationAgent()

        print("All agents initialized successfully.")

    def run(self, customer_data):

        print("\nStarting Customer Intelligence Workflow...")

        # The Recommendation Agent coordinates
        # profiling, prediction and recommendations.
        results = self.recommendation_agent.recommend(
            customer_data
        )

        print("Customer profiling completed.")
        print("Repeat-purchase scoring completed.")
        print("Marketing recommendations completed.")

        return results


# ==========================================
# STEP 3: Run the full workflow
# ==========================================

if __name__ == "__main__":

    orchestrator = OrchestratorAgent()

    # Load historical customer data
    df = pd.read_csv(DATA_PATH)

    # Use April 2018 snapshot
    customers = df[
        df["cutoff_date"] == "2018-04-01"
    ].copy()

    if customers.empty:
        raise ValueError(
            "No customers found for April 2018."
        )

    print("\nCustomers to analyze:", len(customers))

    # Run all agents
    results = orchestrator.run(customers)

    # Select useful output columns
    output_columns = [
        "customer_unique_id",
        "cutoff_date",
        "recency",
        "frequency",
        "customer_tenure",
        "activity_category",
        "purchase_frequency_category",
        "repeat_purchase_score",
        "recommended_action"
    ]

    final_results = results[output_columns].copy()

    # Save results
    output_path = (
        OUTPUT_DIR / "customer_intelligence_results.csv"
    )

    final_results.to_csv(
        output_path,
        index=False
    )

    print("\nCUSTOMER INTELLIGENCE RESULTS")

    print(
        final_results.head(10).to_string(index=False)
    )

    print("\nRECOMMENDATION DISTRIBUTION")

    print(
        final_results[
            "recommended_action"
        ].value_counts()
    )

    print("\nORCHESTRATOR WORKFLOW COMPLETED")
    print("Total customers analyzed:", len(final_results))
    print("Saved to:", output_path)
