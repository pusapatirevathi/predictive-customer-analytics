
import pandas as pd
from pathlib import Path

# =====================================
# STEP 1: Define project paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)


# =====================================
# STEP 2: Customer Profiling Agent
# =====================================

class CustomerProfilingAgent:

    def __init__(self):
        print("Customer Profiling Agent initialized.")

    def classify_activity(self, recency):
        """
        Classify customers based on
        days since their last purchase.
        """

        if recency <= 30:
            return "Recently Active"

        elif recency <= 90:
            return "Moderately Active"

        elif recency <= 180:
            return "At Risk"

        else:
            return "Inactive"

    def profile_customers(self, customer_data):
        """
        Generate customer profiles
        using historical purchase features.
        """

        required_columns = [
            "customer_unique_id",
            "recency",
            "frequency",
            "customer_tenure",
            "orders_last_30d",
            "orders_last_60d",
            "orders_last_90d"
        ]

        missing = [
            col for col in required_columns
            if col not in customer_data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        profiles = customer_data.copy()

        # Assign customer activity category
        profiles["activity_category"] = (
            profiles["recency"].apply(
                self.classify_activity
            )
        )

        # Assign purchasing frequency category
        profiles["purchase_frequency_category"] = (
            profiles["frequency"].apply(
                lambda x: (
                    "Repeat Customer"
                    if x > 1
                    else "One-Time Customer"
                )
            )
        )

        # Calculate recent purchase indicator
        profiles["recent_purchase_flag"] = (
            profiles["orders_last_30d"] > 0
        ).astype(int)

        return profiles


# =====================================
# STEP 3: Test the Profiling Agent
# =====================================

if __name__ == "__main__":

    print("\nStarting Customer Profiling Agent...")

    agent = CustomerProfilingAgent()

    df = pd.read_csv(DATA_PATH)

    # Use the April 2018 historical snapshot
    customers = df[
        df["cutoff_date"] == "2018-04-01"
    ].copy()

    profiles = agent.profile_customers(customers)

    print("\nCUSTOMER PROFILING RESULTS")
    print(
        profiles[
            [
                "customer_unique_id",
                "recency",
                "frequency",
                "activity_category",
                "purchase_frequency_category"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nCUSTOMER ACTIVITY DISTRIBUTION")
    print(
        profiles["activity_category"].value_counts()
    )

    print("\nCUSTOMER PROFILING AGENT COMPLETED")
