import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("chennai max-inventory.csv")

# Page config and styling
st.set_page_config(page_title="Max Inventory Analysis & Restocking Predictor", layout="wide")
st.markdown("""
    <style>
        .main-title {font-size: 36px; font-weight: bold; margin-bottom: 10px;}
        .section-header {font-size: 24px; font-weight: 600; margin-top: 30px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")

# Add 'All' to each filter
categories = ['All'] + sorted(df["Category"].dropna().unique().tolist())
branches = ['All'] + sorted(df["Branch Name"].dropna().unique().tolist())
genders = ['All'] + sorted(df["Gender"].dropna().unique().tolist())
sizes = ['All'] + sorted(df["Size"].dropna().unique().tolist())

selected_category = st.sidebar.selectbox("Category", options=categories)
selected_branch = st.sidebar.selectbox("Branch Name", options=branches)
selected_gender = st.sidebar.selectbox("Gender", options=genders)
selected_size = st.sidebar.selectbox("Size", options=sizes)

# Apply filters dynamically
filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]
if selected_branch != "All":
    filtered_df = filtered_df[filtered_df["Branch Name"] == selected_branch]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
if selected_size != "All":
    filtered_df = filtered_df[filtered_df["Size"] == selected_size]

# Title and Tabs
st.markdown("<div class='main-title'>🛒 Max Inventory Analysis & Restocking Predictor</div>", unsafe_allow_html=True)
tabs = st.tabs(["📊 Dashboard", "🔁 Restocking Predictor"])

# ========== 📊 Dashboard ==========
with tabs[0]:
    st.markdown("<div class='section-header'>📦 Inventory Overview</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(filtered_df))
    col2.metric("Total Available", int(filtered_df["Available Stock"].sum()))
    col3.metric("Total Sold", int(filtered_df["Sold Stock"].sum()))
    avg_price = filtered_df["Price"].mean() if not filtered_df.empty else 0
    col4.metric("Average Price", f"₹{avg_price:.2f}")

    # Bar chart: Sold stock by brand
    if not filtered_df.empty:
        st.markdown(f"<div class='section-header'>📈 Sold Stock by Brand</div>", unsafe_allow_html=True)
        brand_sales = filtered_df.groupby("Brand")["Sold Stock"].sum().reset_index()
        fig = px.bar(brand_sales, x="Brand", y="Sold Stock", color="Brand",
                     title="Sold Stock by Brand", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

# ========== 🔁 Restocking Predictor ==========
with tabs[1]:
    st.markdown("<div class='section-header'>🧮 Selected Item Details</div>", unsafe_allow_html=True)

    if not filtered_df.empty:
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

        st.markdown("<div class='section-header'>📦 Restocking Prediction Results</div>", unsafe_allow_html=True)

        # Restocking prediction logic
        def predict_restock(row):
            threshold = 30
            available = row["Available Stock"]
            restock_needed = "Restock Needed" if available < threshold else "Sufficient"
            quantity = max(0, threshold - available)
            return pd.Series([restock_needed, quantity], index=["Restocking Status", "Quantity to Restock"])

        prediction_df = filtered_df.copy()
        prediction_df[["Restocking Status", "Quantity to Restock"]] = prediction_df.apply(predict_restock, axis=1)

        st.dataframe(prediction_df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No item data available for the selected filters.")
