
# Predictive Customer Analytics Using Machine Learning and Agentic AI

## Overview

This project is an end-to-end customer intelligence platform
developed for a Master's final-semester project in Data Analytics.

It combines customer behavior analysis, machine learning,
LangGraph-based workflow orchestration, and an interactive
Streamlit dashboard.

The system analyzes historical e-commerce transactions,
generates customer profiles, estimates repeat-purchase
ranking scores, and suggests marketing actions.

## Project Objectives

- Analyze customer purchasing behavior.
- Perform RFM analysis and customer segmentation.
- Identify patterns in customer activity.
- Predict repeat purchases within a 90-day window.
- Generate customer-level marketing recommendations.
- Coordinate analytics components using LangGraph.
- Present insights through an interactive dashboard.

## Dataset

**Olist Brazilian E-Commerce Dataset**

The dataset includes customer, order, payment, and
other e-commerce transaction information.

The project uses historical customer transactions
to construct time-based prediction snapshots.

Customer records from April 1, 2018, are used
for the final prediction evaluation and dashboard.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Plotly
- LangGraph
- Streamlit
- Git and GitHub

## Project Architecture

```text
Olist E-Commerce Dataset
          |
          v
Data Cleaning and Preparation
          |
          v
Customer Analytics and Feature Engineering
          |
          v
Machine Learning Prediction Model
          |
          v
LangGraph Workflow
          |
          +-- Customer Profiling Agent
          |
          +-- Prediction Agent
          |
          +-- Recommendation Agent
          |
          +-- Customer Insights Agent
          |
          v
Streamlit Customer Intelligence Dashboard
```

## Machine Learning

### Prediction Objective

Predict whether an existing customer will place
another recorded order within the next 90 days.

### Features

- Recency
- Purchase frequency
- Customer tenure
- Days between first and last purchase
- Orders during the previous 30 days
- Orders during the previous 60 days
- Orders during the previous 90 days

### Algorithms Evaluated

1. Logistic Regression
2. Random Forest

Logistic Regression was selected based on the
historical validation comparison.

### Final Test Results

| Metric | Result |
|---|---:|
| Test customer records | 64,475 |
| Actual repeat purchasers | 438 |
| ROC-AUC | 0.5999 |
| Average Precision | 0.0163 |
| Baseline positive rate | 0.0068 |
| Classification accuracy | 77% |
| Positive-class recall | 38% |

The model demonstrates modest predictive performance.

Because repeat purchases are rare, accuracy alone
does not adequately describe model quality.

Model outputs are ranking scores, not calibrated
purchase probabilities.

## Multi-Agent Workflow

The system contains four specialized components.

### 1. Customer Profiling Agent

Analyzes historical purchasing behavior and
classifies customers into activity categories.

### 2. Prediction Agent

Loads the trained machine learning pipeline
and generates repeat-purchase ranking scores.

### 3. Recommendation Agent

Assigns rule-based marketing recommendations
using customer activity classifications.

### 4. Customer Insights Agent

Summarizes customer activity distributions
and suggested marketing priorities.

### LangGraph Orchestration

LangGraph coordinates the components in a
sequential, stateful workflow.

The current implementation uses deterministic
Python logic rather than autonomous LLM planning.

## Streamlit Dashboard

The dashboard includes:

- Customer overview KPIs
- Customer activity distribution
- Marketing recommendation charts
- Repeat-purchase score distribution
- Customer search and filtering
- Automated customer insights
- Individual customer intelligence
- CSV export

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/pusapatirevathi/predictive-customer-analytics.git

cd predictive-customer-analytics
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib plotly streamlit langgraph joblib
```

### 4. Prepare the dataset

Download the Olist dataset and place the
required CSV files inside `data/raw/`.

The raw dataset is not included in the repository.

### 5. Generate historical customer features

```bash
python analytics/prepare_time_based_data.py
```

### 6. Train and save the prediction model

```bash
python analytics/save_prediction_model.py
```

### 7. Run the LangGraph workflow

```bash
python agents/langgraph_workflow.py
```

### 8. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open the local Streamlit URL shown in the terminal.

## Evaluation and Limitations

- The dataset has a highly imbalanced prediction target.
- Repeat-purchase prediction performance is modest.
- The model is not calibrated.
- Recommendations are heuristic, not experimentally validated.
- Temporal evaluation and historical data availability
  require additional review before production use.
- The system currently uses rule-based agents
  orchestrated by LangGraph.

## Future Improvements

- Improve predictive features and model calibration.
- Evaluate precision among top-ranked customers.
- Add robust time-based cross-validation.
- Introduce LLM-powered reasoning and tool selection.
- Test personalized marketing recommendations.
- Deploy the dashboard for demonstration.

## Conclusion

This project demonstrates a complete customer
analytics workflow integrating machine learning,
customer segmentation, graph-based orchestration,
and an interactive analytics dashboard.

It is intended as an educational prototype
and research demonstration.
