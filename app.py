import streamlit as st
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("chennai max-inventory.csv")

st.set_page_config(page_title="Max Inventory Analysis & Restocking Predictor", layout="wide")
st.markdown("""
    <style>
        .main-title {font-size: 36px; font-weight: bold; margin-bottom: 10px;}
        .section-header {font-size: 24px; font-weight: 600; margin-top: 30px; margin-bottom: 10px;}
        .metric-label {font-weight: 500;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
categories = df["Category"].unique()
branches = df["Branch Name"].unique()
genders = df["Gender"].unique()
sizes = df["Size"].unique()

selected_category = st.sidebar.selectbox("Category", options=categories)
selected_branch = st.sidebar.selectbox("Branch Name", options=branches)
selected_gender = st.sidebar.selectbox("Gender", options=genders)
selected_size = st.sidebar.selectbox("Size", options=sizes)

# Filtered Data
filtered_df = df[
    (df["Category"] == selected_category) &
    (df["Branch Name"] == selected_branch) &
    (df["Gender"] == selected_gender) &
    (df["Size"] == selected_size)
]

# Title
st.markdown("<div class='main-title'>🛒 Max Inventory Analysis & Restocking Predictor</div>", unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["📊 Dashboard", "🔁 Restocking Predictor"])

# ========== 📊 Dashboard ==========
with tabs[0]:
    st.markdown("<div class='section-header'>📦 Inventory Overview</div>", unsafe_allow_html=True)

    # Updated Metrics Based on Filters
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(filtered_df))
    col2.metric("Total Available", int(filtered_df["Available Stock"].sum()))
    col3.metric("Total Sold", int(filtered_df["Sold Stock"].sum()))
    avg_price = filtered_df["Price"].mean() if not filtered_df.empty else 0
    col4.metric("Average Price", f"₹{avg_price:.2f}")

    # Graph - Sold Stock by Brand
    if not filtered_df.empty:
        st.markdown(f"<div class='section-header'>📈 Sold Stock by Brand ({selected_category}, {selected_gender}, {selected_size}, {selected_branch})</div>", unsafe_allow_html=True)
        brand_sales = filtered_df.groupby("Brand")["Sold Stock"].sum().reset_index()
        fig = px.bar(brand_sales, x="Brand", y="Sold Stock", color="Brand",
                     title="Sold Stock by Brand", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

# ========== 🔁 Restocking Predictor ==========
with tabs[1]:
    st.markdown("<div class='section-header'>🧮 Select Item Details</div>", unsafe_allow_html=True)

    # Show filtered table with row selection
    selected_rows = st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    # Show predictions for each row
    if not filtered_df.empty:
        st.markdown("<div class='section-header'>📦 Restocking Prediction Results</div>", unsafe_allow_html=True)
        
        def predict_restock(row):
            threshold = 30  # Set desired threshold
            required_qty = max(0, threshold - row["Available Stock"])
            status = "Restock Needed" if row["Available Stock"] < threshold else "Sufficient"
            return pd.Series([status, required_qty], index=["Restocking Status", "Quantity to Restock"])

        results = filtered_df.apply(predict_restock, axis=1)
        result_df = pd.concat([filtered_df.reset_index(drop=True), results], axis=1)

        st.dataframe(result_df, use_container_width=True)
    else:
        st.info("No item selected or available.")
