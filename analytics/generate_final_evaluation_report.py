
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "final_model_evaluation.md"

# Results from the previously completed final test.
# Update these values if you retrain the model.
results = {
    "test_records": 64475,
    "repeat_purchasers": 438,
    "roc_auc": 0.5999,
    "average_precision": 0.0163,
    "accuracy": 0.77,
    "recall": 0.38,
    "precision": 0.01
}

baseline = (
    results["repeat_purchasers"]
    / results["test_records"]
)

improvement = (
    results["average_precision"] / baseline
)

report = f"""# Final Machine Learning Evaluation Report

## Project Title

Predictive Customer Analytics Using Machine Learning
and Agentic AI

## 1. Project Objective

The objective is to analyze historical customer behavior
and predict whether an existing customer will make another
purchase within the next 90 days.

## 2. Dataset

Olist Brazilian E-Commerce Dataset.

Historical customer transactions were used to engineer
customer purchase activity features.

## 3. Machine Learning Model

Logistic Regression with feature scaling and class
weighting.

## 4. Evaluation Results

| Metric | Value |
|---|---:|
| Test customer records | {results['test_records']:,} |
| Repeat purchasers | {results['repeat_purchasers']:,} |
| ROC-AUC | {results['roc_auc']:.4f} |
| Average Precision | {results['average_precision']:.4f} |
| Baseline positive rate | {baseline:.4f} |
| Accuracy | {results['accuracy']:.2%} |
| Repeat-purchase recall | {results['recall']:.2%} |
| Repeat-purchase precision | Approximately {results['precision']:.2%} |

## 5. Model Interpretation

The dataset is highly imbalanced.

Only {baseline:.2%} of test customer records
represent repeat purchasers.

The model achieved an average precision of
{results['average_precision']:.4f}, approximately
{improvement:.2f} times the random-ranking baseline.

This indicates limited predictive signal beyond
random ranking.

Accuracy is not an appropriate standalone measure
because the negative class dominates the dataset.

## 6. Model Limitations

- Very few repeat-purchase events are available.
- Predictive performance is modest.
- False-positive predictions remain frequent.
- Model scores are not calibrated probabilities.
- Historical order data may not capture all factors
  affecting customer purchase behavior.
- Recommendations have not been tested in
  controlled marketing experiments.

## 7. Future Improvements

- Evaluate precision at top-ranked customer groups.
- Test additional customer behavior features.
- Tune classification thresholds using validation data.
- Evaluate probability calibration.
- Compare models using repeated temporal evaluations.
- Test marketing recommendations through experiments.

## 8. Conclusion

The project demonstrates an end-to-end system
combining customer analytics, machine learning,
LangGraph orchestration, and a Streamlit dashboard.

The predictive model should be considered
experimental rather than production-ready.
"""

OUTPUT_PATH.write_text(report, encoding="utf-8")

print("FINAL MODEL EVALUATION REPORT GENERATED")
print("Saved to:", OUTPUT_PATH)

print("\nMODEL SUMMARY")
print("Test records:", results["test_records"])
print("Repeat purchasers:", results["repeat_purchasers"])
print("ROC-AUC:", results["roc_auc"])
print("Average precision:", results["average_precision"])
print("Baseline positive rate:", round(baseline, 4))
print("Improvement over baseline:", round(improvement, 2))
