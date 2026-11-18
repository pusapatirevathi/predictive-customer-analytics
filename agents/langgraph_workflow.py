
import sys
from pathlib import Path
from typing import TypedDict

import pandas as pd
from langgraph.graph import StateGraph, START, END

# =====================================
# Project paths
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "agents"))

from customer_profiling_agent import CustomerProfilingAgent
from prediction_agent import PredictionAgent
from recommendation_agent import RecommendationAgent
from insights_agent import CustomerInsightsAgent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_time_based_data.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed"
    / "langgraph_customer_results.csv"
)

# =====================================
# Define shared workflow state
# =====================================

class CustomerState(TypedDict):
    customers: pd.DataFrame
    results: pd.DataFrame
    insights: list[str]


# =====================================
# Initialize agents
# =====================================

profiling_agent = CustomerProfilingAgent()
prediction_agent = PredictionAgent()
recommendation_agent = RecommendationAgent()
insights_agent = CustomerInsightsAgent()


# =====================================
# Node 1: Customer Profiling
# =====================================

def profiling_node(state: CustomerState):
    print("Running Customer Profiling Agent...")

    profiles = profiling_agent.profile_customers(
        state["customers"]
    )

    return {"results": profiles}


# =====================================
# Node 2: Prediction
# =====================================

def prediction_node(state: CustomerState):
    print("Running Prediction Agent...")

    predictions = prediction_agent.predict(
        state["results"]
    )

    return {"results": predictions}


# =====================================
# Node 3: Recommendations
# =====================================

def recommendation_node(state: CustomerState):
    print("Running Recommendation Agent...")

    results = state["results"].copy()

    results["recommended_action"] = results.apply(
        recommendation_agent.generate_recommendation,
        axis=1
    )

    return {"results": results}


# =====================================
# Node 4: Customer Insights
# =====================================

def insights_node(state: CustomerState):
    print("Running Customer Insights Agent...")

    report = insights_agent.generate_insights(
        state["results"]
    )

    return {"insights": report["insights"]}


# =====================================
# Build LangGraph
# =====================================

workflow = StateGraph(CustomerState)

workflow.add_node("profiling", profiling_node)
workflow.add_node("prediction", prediction_node)
workflow.add_node("recommendation", recommendation_node)
workflow.add_node("insights", insights_node)

workflow.add_edge(START, "profiling")
workflow.add_edge("profiling", "prediction")
workflow.add_edge("prediction", "recommendation")
workflow.add_edge("recommendation", "insights")
workflow.add_edge("insights", END)

app = workflow.compile()


# =====================================
# Run workflow
# =====================================

if __name__ == "__main__":

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing customer data. Run "
            "python analytics/prepare_time_based_data.py"
        )

    df = pd.read_csv(DATA_PATH)

    customers = df[
        df["cutoff_date"] == "2018-04-01"
    ].copy()

    if customers.empty:
        raise ValueError(
            "No customers found for April 2018."
        )

    print("Starting LangGraph Workflow...")
    print("Customers:", len(customers))

    initial_state = {
        "customers": customers,
        "results": pd.DataFrame(),
        "insights": []
    }

    final_state = app.invoke(initial_state)

    results = final_state["results"]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nLANGGRAPH CUSTOMER INSIGHTS")

    for insight in final_state["insights"]:
        print("-", insight)

    print("\nLANGGRAPH WORKFLOW COMPLETED")
    print("Customers processed:", len(results))
    print("Saved to:", OUTPUT_PATH)
