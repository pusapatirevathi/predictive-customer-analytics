
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# ======================================
# STEP 1: Project paths and agent import
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "agents"))

from insights_agent import CustomerInsightsAgent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_intelligence_results.csv"
)

# ======================================
# STEP 2: Page configuration
# ======================================

st.set_page_config(
    page_title="Predictive Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Predictive Customer Analytics")

st.subheader(
    "Machine Learning & Multi-Agent Customer Intelligence"
)

st.markdown(
    "Analyze customer behavior, repeat-purchase scores, "
    "marketing recommendations, and automated insights."
)

# ======================================
# STEP 3: Load customer intelligence data
# ======================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not DATA_PATH.exists():
    st.error(
        "Customer intelligence data not found. "
        "Run: python agents/orchestrator_agent.py"
    )
    st.stop()

df = load_data()

# ======================================
# STEP 4: Sidebar filters
# ======================================

st.sidebar.header("Customer Filters")

activities = st.sidebar.multiselect(
    "Activity Category",
    options=sorted(df["activity_category"].unique()),
    default=sorted(df["activity_category"].unique())
)

filtered_df = df[
    df["activity_category"].isin(activities)
].copy()

# ======================================
# STEP 5: Customer overview KPIs
# ======================================

st.header("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Recency",
    (
        f"{filtered_df['recency'].mean():.1f} days"
        if not filtered_df.empty
        else "N/A"
    )
)

col3.metric(
    "Average Purchase Frequency",
    (
        f"{filtered_df['frequency'].mean():.2f}"
        if not filtered_df.empty
        else "N/A"
    )
)

col4.metric(
    "Average Prediction Score",
    (
        f"{filtered_df['repeat_purchase_score'].mean():.3f}"
        if not filtered_df.empty
        else "N/A"
    )
)

st.caption(
    "Prediction scores are model ranking outputs, "
    "not calibrated probabilities of repeat purchase."
)

if filtered_df.empty:
    st.warning(
        "No customers match the selected filters."
    )
    st.stop()

# ======================================
# STEP 6: Customer activity analysis
# ======================================

st.header("Customer Activity Analysis")

activity_counts = (
    filtered_df["activity_category"]
    .value_counts()
    .reset_index()
)

activity_counts.columns = [
    "Activity Category",
    "Customers"
]

fig_activity = px.bar(
    activity_counts,
    x="Activity Category",
    y="Customers",
    color="Activity Category",
    title="Customer Activity Distribution"
)

st.plotly_chart(
    fig_activity,
    use_container_width=True
)

# ======================================
# STEP 7: Marketing recommendations
# ======================================

st.header("Marketing Recommendations")

recommendation_counts = (
    filtered_df["recommended_action"]
    .value_counts()
    .reset_index()
)

recommendation_counts.columns = [
    "Recommended Action",
    "Customers"
]

fig_recommendations = px.pie(
    recommendation_counts,
    names="Recommended Action",
    values="Customers",
    title="Recommended Marketing Actions",
    hole=0.4
)

st.plotly_chart(
    fig_recommendations,
    use_container_width=True
)

st.caption(
    "Marketing actions are rule-based suggestions "
    "and have not been validated through experiments."
)

# ======================================
# STEP 8: Prediction score distribution
# ======================================

st.header("Repeat-Purchase Prediction Scores")

fig_scores = px.histogram(
    filtered_df,
    x="repeat_purchase_score",
    nbins=30,
    title="Distribution of Repeat-Purchase Scores"
)

st.plotly_chart(
    fig_scores,
    use_container_width=True
)

# ======================================
# STEP 9: Customer Intelligence Explorer
# ======================================

st.header("Customer Intelligence Explorer")

search_id = st.text_input(
    "Search by Customer ID"
)

display_df = filtered_df.copy()

if search_id:
    display_df = display_df[
        display_df["customer_unique_id"]
        .astype(str)
        .str.contains(
            search_id,
            case=False,
            regex=False
        )
    ]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ======================================
# STEP 10: Export customer results
# ======================================

st.download_button(
    label="Download Filtered Customer Results",
    data=display_df.to_csv(index=False),
    file_name="customer_intelligence_export.csv",
    mime="text/csv"
)

# ======================================
# STEP 11: Automated Customer Insights
# ======================================

st.divider()

st.header("Automated Customer Insights")

st.markdown(
    "The Customer Insights Agent analyzes customer "
    "behavior and generates data-driven summaries "
    "and suggested marketing priorities."
)

# Initialize the Insights Agent
insights_agent = CustomerInsightsAgent()

# Analyze currently filtered customer data
insights_results = insights_agent.generate_insights(
    filtered_df
)

# ======================================
# STEP 12: Display key customer insights
# ======================================

st.subheader("Key Customer Insights")

for insight in insights_results["insights"]:
    st.info(insight)

# ======================================
# STEP 13: Customer activity summary
# ======================================

st.subheader("Customer Activity Summary")

activity_summary = (
    insights_results["activity_distribution"]
    .reset_index()
)

activity_summary.columns = [
    "Activity Category",
    "Customer Count"
]

st.dataframe(
    activity_summary,
    use_container_width=True,
    hide_index=True
)

# ======================================
# STEP 14: Marketing action summary
# ======================================

st.subheader("Recommended Marketing Actions")

recommendation_summary = (
    insights_results["recommendation_distribution"]
    .reset_index()
)

recommendation_summary.columns = [
    "Recommended Action",
    "Customer Count"
]

st.dataframe(
    recommendation_summary,
    use_container_width=True,
    hide_index=True
)

# ======================================
# STEP 15: Dashboard completion
# ======================================

st.success(
    "Customer Insights Agent analysis completed."
)

st.divider()

st.caption(
    "Predictive Customer Analytics | "
    "Machine Learning & Multi-Agent AI Project"
)

# ==========================================
# STEP 16: Individual Customer Intelligence
# ==========================================

st.divider()

st.header("Individual Customer Intelligence")

st.markdown(
    "Select a customer to view their purchase "
    "behavior, prediction score, and recommended "
    "marketing action."
)

# Use customers from the current filters
customer_ids = sorted(
    filtered_df["customer_unique_id"]
    .astype(str)
    .unique()
    .tolist()
)

selected_customer = st.selectbox(
    "Select Customer ID",
    options=customer_ids
)

customer = filtered_df[
    filtered_df["customer_unique_id"].astype(str)
    == selected_customer
].iloc[0]

st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Purchase Frequency",
    int(customer["frequency"])
)

col2.metric(
    "Recency",
    f"{int(customer['recency'])} days"
)

col3.metric(
    "Customer Tenure",
    f"{int(customer['customer_tenure'])} days"
)

st.subheader("Customer Prediction")

st.metric(
    "Repeat-Purchase Model Score",
    f"{customer['repeat_purchase_score']:.3f}"
)

st.caption(
    "This is a model ranking score, "
    "not a calibrated purchase probability."
)

st.subheader("Customer Activity Category")

st.info(
    customer["activity_category"]
)

st.subheader("Recommended Marketing Action")

st.success(
    customer["recommended_action"]
)

st.caption(
    "Marketing recommendations are rule-based "
    "suggestions and have not been experimentally "
    "validated."
)
