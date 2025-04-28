import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("chennai max-inventory.csv")

df = load_data()

# Title
st.title("🛍️ Chennai Max Inventory Dashboard")

# Sidebar Filters
st.sidebar.header("Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique())

# Apply filters
filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products", len(filtered_df))
col3.metric("Total Available", filtered_df["Available Stock"].sum())
col3.metric("Total Sold", filtered_df["Sold Stock"].sum())
col4.metric("Average Price", f"₹{filtered_df['Price'].mean():.2f}")

# ----------------------------
# Restocking Prediction Section
# ----------------------------
st.markdown("---")
st.header("📦 Restocking Predictor")

col_a, col_b = st.columns(2)

with col_a:
    available_stock = st.number_input("Enter Available Stock", min_value=0, step=1)
with col_b:
    sold_stock = st.number_input("Enter Sold Stock", min_value=0, step=1)

if available_stock or sold_stock:
    restock_needed = "Yes" if available_stock < sold_stock else "No"
    status_color = "green" if restock_needed == "Yes" else "red"
    st.markdown(f"### ✅ Restocking Status: <span style='color:{status_color}'>{restock_needed}</span>", unsafe_allow_html=True)

# Plot
st.subheader("Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax)
st.pyplot(fig)

# Show data
st.subheader("Filtered Data")
st.dataframe(filtered_df)
