
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_intelligence_results.csv"
)


class CustomerInsightsAgent:

    def __init__(self):
        print("Customer Insights Agent initialized.")

    def generate_insights(self, customer_data):

        if customer_data.empty:
            raise ValueError("Customer dataset is empty.")

        required_columns = [
            "activity_category",
            "repeat_purchase_score",
            "recommended_action"
        ]

        missing = [
            col for col in required_columns
            if col not in customer_data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        total_customers = len(customer_data)

        activity_counts = (
            customer_data["activity_category"]
            .value_counts()
        )

        recommendation_counts = (
            customer_data["recommended_action"]
            .value_counts()
        )

        average_score = (
            customer_data["repeat_purchase_score"]
            .mean()
        )

        inactive_count = int(
            activity_counts.get("Inactive", 0)
        )

        at_risk_count = int(
            activity_counts.get("At Risk", 0)
        )

        recently_active_count = int(
            activity_counts.get("Recently Active", 0)
        )

        inactive_percentage = (
            inactive_count / total_customers * 100
        )

        at_risk_percentage = (
            at_risk_count / total_customers * 100
        )

        insights = [
            (
                f"Analyzed {total_customers:,} "
                "customer records."
            ),
            (
                f"{inactive_count:,} customers "
                f"({inactive_percentage:.1f}%) "
                "are classified as inactive."
            ),
            (
                f"{at_risk_count:,} customers "
                f"({at_risk_percentage:.1f}%) "
                "are classified as at risk."
            ),
            (
                f"{recently_active_count:,} customers "
                "are recently active."
            ),
            (
                f"The average repeat-purchase model "
                f"score is {average_score:.3f}. "
                "This is not a calibrated probability."
            ),
            (
                "Inactive customers may be considered "
                "for a win-back campaign."
            ),
            (
                "At-risk customers may be considered "
                "for re-engagement campaigns."
            )
        ]

        return {
            "total_customers": total_customers,
            "average_score": average_score,
            "activity_distribution": activity_counts,
            "recommendation_distribution": recommendation_counts,
            "insights": insights
        }


if __name__ == "__main__":

    print("\nStarting Customer Insights Agent...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Customer intelligence results not found. "
            "Run python agents/orchestrator_agent.py"
        )

    df = pd.read_csv(DATA_PATH)

    agent = CustomerInsightsAgent()

    results = agent.generate_insights(df)

    print("\nCUSTOMER INTELLIGENCE INSIGHTS")
    print("==============================")

    for index, insight in enumerate(
        results["insights"], start=1
    ):
        print(f"{index}. {insight}")

    print("\nCUSTOMER INSIGHTS AGENT COMPLETED")
