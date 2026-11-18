# Final Machine Learning Evaluation Report

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
| Test customer records | 64,475 |
| Repeat purchasers | 438 |
| ROC-AUC | 0.5999 |
| Average Precision | 0.0163 |
| Baseline positive rate | 0.0068 |
| Accuracy | 77.00% |
| Repeat-purchase recall | 38.00% |
| Repeat-purchase precision | Approximately 1.00% |

## 5. Model Interpretation

The dataset is highly imbalanced.

Only 0.68% of test customer records
represent repeat purchasers.

The model achieved an average precision of
0.0163, approximately
2.40 times the random-ranking baseline.

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
