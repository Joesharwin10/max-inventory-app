import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Max Showroom Data.csv")

df = load_data()

# Data preprocessing (consistent with notebook)
df["Total Items"] = df["Available Stock"] + df["Sold Stock"]
df["Target"] = df["Available Stock"] < df["Sold Stock"]
df["Target"] = df["Target"].astype(int)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
categories = ['All'] + sorted(df["Category"].dropna().unique())
branches = ['All'] + sorted(df["Branch Name"].dropna().unique())
genders = ['All'] + sorted(df["Gender"].dropna().unique())
brands = ['All'] + sorted(df["Brand"].dropna().unique())
sizes = ['All'] + sorted(df["Size"].dropna().unique())

selected_category = st.sidebar.selectbox("Category", categories)
selected_branch = st.sidebar.selectbox("Branch Name", branches)
selected_gender = st.sidebar.selectbox("Gender", genders)
selected_brand = st.sidebar.selectbox("Brand", brands)
selected_size = st.sidebar.selectbox("Size", sizes)

# Filter data
filtered_df = df.copy()
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]
if selected_branch != 'All':
    filtered_df = filtered_df[filtered_df['Branch Name'] == selected_branch]
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
if selected_brand != 'All':
    filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
if selected_size != 'All':
    filtered_df = filtered_df[filtered_df['Size'] == selected_size]

# Apply styles
st.markdown("""
    <style>
        .main-title {font-size: 36px; font-weight: bold; margin-bottom: 10px;}
        .section-header {font-size: 24px; font-weight: 600; margin-top: 30px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-title'>🛒 Max Inventory Analysis & ML-Based Restocking Predictor</div>", unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["📊 Dashboard", "🔁 Restocking Predictor"])

# ========== 📊 Dashboard ==========
with tabs[0]:
    st.markdown("<div class='section-header'>📦 Inventory Overview</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", len(filtered_df))
    col2.metric("Available Stock", int(filtered_df["Available Stock"].sum()))
    col3.metric("Sold Stock", int(filtered_df["Sold Stock"].sum()))
    avg_price = filtered_df["Price"].mean() if not filtered_df.empty else 0
    col4.metric("Avg. Price", f"₹{avg_price:.2f}")

    if not filtered_df.empty:
        st.markdown("<div class='section-header'>📈 Sold Stock by Brand</div>", unsafe_allow_html=True)
        brand_chart = filtered_df.groupby("Brand")["Sold Stock"].sum().reset_index()
        fig = px.bar(brand_chart, x="Brand", y="Sold Stock", color="Brand", title="Sold Stock by Brand", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected filters.")

# ========== 🔁 Restocking Predictor ==========
with tabs[1]:
    st.markdown("<div class='section-header'>📦 Restocking Prediction (ML-Based)</div>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Prepare training data
        features = ["Available Stock", "Sold Stock", "Total Items", "Price"]
        df_model = df.copy()

        scaler = MinMaxScaler()
        df_model[["Price"]] = scaler.fit_transform(df_model[["Price"]])

        X = df_model[features]
        y = df_model["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Apply same scaling to filtered data
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy["Total Items"] = filtered_df_copy["Available Stock"] + filtered_df_copy["Sold Stock"]
        filtered_df_copy[["Price"]] = scaler.transform(filtered_df_copy[["Price"]])

        # Predict
        X_filtered = filtered_df_copy[features]
        predictions = model.predict(X_filtered)

        result_df = filtered_df.copy()
        result_df["Restock Needed"] = ["Yes" if p == 1 else "No" for p in predictions]

        # Show result table
        st.dataframe(result_df[["Branch Name", "Category", "Brand", "Available Stock", "Sold Stock", "Restock Needed"]], use_container_width=True)

    else:
        st.info("No data to display. Please adjust your filters.")
