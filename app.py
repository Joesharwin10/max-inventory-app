import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Max Inventory Dashboard", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Max Showroom Data.csv")
    df.dropna(subset=["Available Stock", "Sold Stock", "Total Items", "Target", "Price", "Restock Needed"], inplace=True)

    # Normalize the Price column
    scaler = MinMaxScaler()
    df["Price"] = scaler.fit_transform(df[["Price"]])

    return df

df = load_data()

# ----------------------------
# Sidebar Filter Section
# ----------------------------
st.title("🛍️ Chennai Max Inventory Dashboard")

st.sidebar.header("Filter Options")
category = st.sidebar.multiselect("Category", options=df["Category"].unique())
gender = st.sidebar.multiselect("Gender", options=df["Gender"].unique())

filtered_df = df.copy()
if category:
    filtered_df = filtered_df[filtered_df["Category"].isin(category)]
if gender:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

# ----------------------------
# Metrics
# ----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", len(filtered_df))
col2.metric("Total Sold", filtered_df["Sold Stock"].sum())
col3.metric("Average Price", f"₹{filtered_df['Price'].mean():.2f}")

# ----------------------------
# Model Training (from notebook logic)
# ----------------------------
X = df[["Available Stock", "Sold Stock", "Total Items", "Target", "Price"]]
y = df["Restock Needed"].apply(lambda x: 1 if x > 0 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Restocking Predictor Section
# ----------------------------
st.markdown("---")
st.header("📦 Restocking Predictor (ML Model)")

col_a, col_b, col_c = st.columns(3)
with col_a:
    available_stock = st.number_input("Available Stock", min_value=0)
with col_b:
    sold_stock = st.number_input("Sold Stock", min_value=0)
with col_c:
    total_items = st.number_input("Total Items", min_value=0)

target = 100  # default business target assumption
price = df["Price"].mean()  # average normalized price

if st.button("Predict Restocking Need"):
    input_data = pd.DataFrame([[available_stock, sold_stock, total_items, target, price]],
                              columns=["Available Stock", "Sold Stock", "Total Items", "Target", "Price"])
    prediction = model.predict(input_data)[0]
    restock_needed = "Yes" if prediction == 1 else "No"
    color = "green" if prediction == 1 else "red"
    st.markdown(f"### ✅ Restocking Status: <span style='color:{color}'>{restock_needed}</span>", unsafe_allow_html=True)

# ----------------------------
# Plot Section
# ----------------------------
st.subheader("Sold Stock by Size")
plot_df = filtered_df.groupby("Size")["Sold Stock"].sum()
fig, ax = plt.subplots()
plot_df.plot(kind="bar", ax=ax, color="skyblue")
ax.set_ylabel("Sold Stock")
ax.set_title("Sold Stock by Size")
st.pyplot(fig)

# ----------------------------
# Data Table
# ----------------------------
st.subheader("Filtered Data")
st.dataframe(filtered_df)
