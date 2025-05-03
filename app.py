import streamlit as st
 import pandas as pd
 import matplotlib.pyplot as plt
 import plotly.express as px
 
 st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")
 # Load dataset
 df = pd.read_csv("chennai max-inventory.csv")
 
 # Load data
 @st.cache_data
 def load_data():
     return pd.read_csv("chennai max-inventory.csv")
 # Page config and styling
 st.set_page_config(page_title="Max Inventory Analysis & Restocking Predictor", layout="wide")
 st.markdown("""
     <style>
         .main-title {font-size: 36px; font-weight: bold; margin-bottom: 10px;}
         .section-header {font-size: 24px; font-weight: 600; margin-top: 30px; margin-bottom: 10px;}
     </style>
 """, unsafe_allow_html=True)
 
 df = load_data()
 # Sidebar Filters
 st.sidebar.header("🔍 Filter Options")
 
 # Title
 st.title("🛍️ Chennai Max Inventory Dashboard")
 # Add 'All' to each filter
 categories = ['All'] + sorted(df["Category"].dropna().unique().tolist())
 branches = ['All'] + sorted(df["Branch Name"].dropna().unique().tolist())
 genders = ['All'] + sorted(df["Gender"].dropna().unique().tolist())
 sizes = ['All'] + sorted(df["Size"].dropna().unique().tolist())
 
 # Sidebar Filters
 st.sidebar.header("Filter Options")
 category = st.sidebar.multiselect("Category", options=df["Category"].unique())
 Branch_Name = st.sidebar.multiselect("Branch Name", options=df["Branch Name"].unique())
 gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique())
 selected_category = st.sidebar.selectbox("Category", options=categories)
 selected_branch = st.sidebar.selectbox("Branch Name", options=branches)
 selected_gender = st.sidebar.selectbox("Gender", options=genders)
 selected_size = st.sidebar.selectbox("Size", options=sizes)
 
 # Apply filters
 # Apply filters dynamically
 filtered_df = df.copy()
 if category:
     filtered_df = filtered_df[filtered_df["Category"].isin(category)]
 if Branch_Name:
     filtered_df = filtered_df[filtered_df["Branch Name"].isin(Branch_Name)]
 
 if gender:
     filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]
 
 # Metrics
 col1, col2, col3, col4 = st.columns(4)
 col1.metric("Total Products", len(filtered_df))
 col2.metric("Total Available", filtered_df["Available Stock"].sum())
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
