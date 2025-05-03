import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("Max Showroom Data.csv")

df = load_data()

# --- Page Setup ---
st.set_page_config(layout="wide")
st.title("🛍️ Max Showroom Inventory Dashboard & Restocking Predictor")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Options")

category = st.sidebar.selectbox("Category", sorted(df["Category"].unique()))
branch = st.sidebar.selectbox("Branch Name", sorted(df["Branch Name"].unique()))
gender = st.sidebar.selectbox("Gender", sorted(df["Gender"].unique()))
size = st.sidebar.selectbox("Size", sorted(df["Size"].unique()))
brand = st.sidebar.selectbox("Brand", sorted(df["Brand"].unique()))

# Apply Filters
filtered_df = df[
    (df["Category"] == category) &
    (df["Branch Name"] == branch) &
    (df["Gender"] == gender) &
    (df["Size"] == size) &
    (df["Brand"] == brand)
]

# --- Inventory Overview ---
st.markdown("### 📦 Inventory Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Available", int(filtered_df["Available Stock"].sum()))
col3.metric("Total Sold", int(filtered_df["Sold Stock"].sum()))
avg_price = filtered_df["Price"].mean() if not filtered_df.empty else 0
col4.metric("Average Price", f"₹{avg_price:.2f}")

# --- Plot Sold Stock by Brand ---
st.markdown("### 📊 Sold Stock by Brand (Filtered)")

if not filtered_df.empty:
    chart_df = filtered_df.groupby("Brand")["Sold Stock"].sum().reset_index()
    fig = px.bar(chart_df, x="Brand", y="Sold Stock", color="Brand", text="Sold Stock")
    fig.update_layout(title_x=0.5, xaxis_title="Brand", yaxis_title="Sold Stock")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data found for selected filters.")

# --- Restocking Predictor ---
st.markdown("### 🤖 Restocking Predictor")

if not filtered_df.empty:
    selected_index = st.selectbox(
        "Select a Product:",
        filtered_df.index,
        format_func=lambda x: f"{filtered_df.loc[x, 'Brand']} - {filtered_df.loc[x, 'Size']} (Avail: {filtered_df.loc[x, 'Available Stock']}, Sold: {filtered_df.loc[x, 'Sold Stock']})"
    )

    row = filtered_df.loc[selected_index]

    # ML model on full dataset
    train_df = df[df["Available Stock"] > 0][["Available Stock", "Sold Stock"]]
    X = train_df[["Available Stock"]]
    y = train_df["Sold Stock"]

    model = LinearRegression()
    model.fit(X, y)

    predicted_sold = model.predict([[row["Available Stock"]]])[0]
    restock_qty = max(0, int(predicted_sold - row["Available Stock"]))

    st.dataframe(pd.DataFrame(row).T)

    if restock_qty > 0:
        st.success(f"🔁 Restocking Needed: {restock_qty} units")
    else:
        st.info("✅ Stock is Sufficient")
else:
    st.info("Please refine your filters to view prediction.")
