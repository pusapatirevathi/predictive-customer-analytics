
# Machine Learning Model Evaluation

## Project
Predictive Customer Analytics Using Machine Learning and Agentic AI

## Objective
Predict whether an existing customer will make another
purchase within the next 90 days using historical
customer purchase behavior.

## Dataset
Olist Brazilian E-Commerce Dataset

The project uses historical customer order and payment
records to engineer purchasing behavior features.

## Model
Logistic Regression with feature scaling and
class weighting to address class imbalance.

## Features
- Recency
- Frequency
- Customer tenure
- Days between first and last purchase
- Orders in the last 30 days
- Orders in the last 60 days
- Orders in the last 90 days

## Evaluation Approach
Historical time-based training and testing were used
instead of relying only on a random train/test split.

The final test snapshot was April 1, 2018.

## Final Test Results

| Metric | Result |
|---|---:|
| Test customer records | 64,475 |
| Actual repeat purchasers | 438 |
| Repeat-purchase prevalence | 0.68% |
| ROC-AUC | 0.5999 |
| Average Precision | 0.0163 |
| Classification Accuracy | 77% |
| Positive-class Recall | 38% |
| Positive-class Precision | Approximately 1% |

## Interpretation

The repeat-purchase prediction problem is highly
imbalanced, with fewer than 1% of customer records
having a positive label.

The model achieves an average precision of 0.0163,
compared with a positive-class baseline of
approximately 0.0068.

This indicates some ranking ability beyond a random
baseline, but overall predictive performance remains
modest.

Classification accuracy is not a sufficient measure
of success because the majority of customers do not
make a repeat purchase within the prediction window.

## Limitations

- Repeat purchases are rare in the dataset.
- Historical purchasing behavior provides limited
  predictive information.
- Prediction scores are not calibrated probabilities.
- Marketing recommendations are rule-based.
- The project has not demonstrated that recommended
  campaigns increase customer retention.
- Historical data availability and potential
  selection bias should be reviewed before
  production use.

## Future Improvements

- Evaluate precision and recall at the top-ranked
  customer percentages.
- Explore additional historical customer features.
- Tune decision thresholds using validation data.
- Test model calibration.
- Evaluate recommendations using controlled
  marketing experiments.
- Integrate LangGraph or an LLM for more advanced
  agent orchestration.

## Conclusion

The project demonstrates an end-to-end customer
analytics system that combines customer profiling,
machine learning predictions, marketing
recommendations, and an interactive dashboard.

The model should currently be treated as an
experimental customer-ranking component rather
than a production-ready prediction system.
