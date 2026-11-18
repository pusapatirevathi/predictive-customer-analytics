
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "langgraph_customer_results.csv"
)

def main():
    print("=" * 55)
    print("CUSTOMER INTELLIGENCE VALIDATION")
    print("=" * 55)

    if not DATA_PATH.exists():
        print("FAIL: LangGraph results file not found.")
        return

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "customer_unique_id",
        "recency",
        "frequency",
        "customer_tenure",
        "activity_category",
        "repeat_purchase_score",
        "recommended_action"
    ]

    checks = {}

    checks["Required columns"] = all(
        col in df.columns for col in required_columns
    )

    checks["Dataset not empty"] = len(df) > 0

    if checks["Required columns"] and not df.empty:

        checks["No missing values"] = (
            df[required_columns].notna().all().all()
        )

        checks["No duplicate customer IDs"] = (
            not df["customer_unique_id"].duplicated().any()
        )

        checks["Valid prediction scores"] = (
            pd.to_numeric(
                df["repeat_purchase_score"],
                errors="coerce"
            ).between(0, 1).all()
        )

        checks["Valid purchase frequency"] = (
            pd.to_numeric(
                df["frequency"],
                errors="coerce"
            ).ge(1).all()
        )

        checks["Valid recency"] = (
            pd.to_numeric(
                df["recency"],
                errors="coerce"
            ).ge(0).all()
        )

        checks["Valid customer tenure"] = (
            pd.to_numeric(
                df["customer_tenure"],
                errors="coerce"
            ).ge(0).all()
        )

        checks["Recommendations populated"] = (
            df["recommended_action"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        )

    print("\nVALIDATION RESULTS\n")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    print("\nTotal customer records:", len(df))

    if "activity_category" in df.columns:
        print("\nCustomer Activity Distribution:")
        print(df["activity_category"].value_counts())

    if "repeat_purchase_score" in df.columns:
        print("\nPrediction Score Summary:")
        print(df["repeat_purchase_score"].describe())

    if all(checks.values()):
        print("\nALL DATA VALIDATION CHECKS PASSED")
    else:
        print("\nSOME DATA VALIDATION CHECKS FAILED")

if __name__ == "__main__":
    main()

