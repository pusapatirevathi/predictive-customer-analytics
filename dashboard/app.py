
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ======================================
# STEP 1: Page configuration
# ======================================

st.set_page_config(
    page_title="Predictive Customer Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Predictive Customer Analytics")
st.subheader("Machine Learning & Multi-Agent Customer Intelligence")

st.markdown(
    "Analyze customer behavior, repeat-purchase scores, "
    "and recommended marketing actions."
)

# ======================================
# STEP 2: Load customer intelligence data
# ======================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed"
    / "customer_intelligence_results.csv"
)

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
# STEP 3: Sidebar filters
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
# STEP 4: Dashboard KPIs
# ======================================

st.header("Customer Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average Recency",
    f"{filtered_df['recency'].mean():.1f} days"
    if not filtered_df.empty else "N/A"
)

col3.metric(
    "Average Purchase Frequency",
    f"{filtered_df['frequency'].mean():.2f}"
    if not filtered_df.empty else "N/A"
)

col4.metric(
    "Average Prediction Score",
    f"{filtered_df['repeat_purchase_score'].mean():.3f}"
    if not filtered_df.empty else "N/A"
)

if filtered_df.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# ======================================
# STEP 5: Customer activity chart
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

st.plotly_chart(fig_activity, use_container_width=True)

# ======================================
# STEP 6: Recommendation distribution
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

# ======================================
# STEP 7: Prediction score distribution
# ======================================

st.header("Repeat-Purchase Prediction Scores")

st.caption(
    "Scores are model ranking outputs, "
    "not calibrated purchase probabilities."
)

fig_scores = px.histogram(
    filtered_df,
    x="repeat_purchase_score",
    nbins=30,
    title="Distribution of Repeat-Purchase Scores"
)

st.plotly_chart(fig_scores, use_container_width=True)

# ======================================
# STEP 8: Customer search
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
# STEP 9: Export results
# ======================================

st.download_button(
    label="Download Filtered Customer Results",
    data=display_df.to_csv(index=False),
    file_name="customer_intelligence_export.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Predictive Customer Analytics | "
    "Machine Learning & Multi-Agent AI Project"
)
