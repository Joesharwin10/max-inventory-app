import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and clean the data
df = pd.read_csv("Max Showroom Data.csv")
df = df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"])

# Prepare features and target
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)

# Train model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X, y)

# Streamlit config
st.set_page_config(page_title="Restocking Predictor", layout="wide")
st.title("🛍️ Max Inventory Restocking Predictor")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
categories = ['All'] + sorted(df["Category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", options=categories)

filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

st.subheader("📦 Filtered Inventory Data")
st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# Predict restocking status
st.subheader("🔁 Restocking Prediction")
if not filtered_df.empty:
    pred_X = filtered_df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
    pred_y = model.predict(pred_X)
    filtered_df["Restocking Status"] = pred_y
    filtered_df["Restocking Status"] = filtered_df["Restocking Status"].map({1: "Restock Needed", 0: "Sufficient"})

    def highlight_status(val):
        color = '#ffcccc' if val == "Restock Needed" else '#ccffcc'
        return f'background-color: {color}'

    styled = filtered_df[["Brand", "Available Stock", "Sold Stock", "Target", "Price", "Restocking Status"]]
    st.dataframe(styled.style.applymap(highlight_status, subset=["Restocking Status"]), use_container_width=True)
else:
    st.info("No data to show based on selected filters.")
