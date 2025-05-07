import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Max Inventory 2025 Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Max 2023-2024.csv")

df = load_data()

# Preprocessing
df["Total Items"] = df["Available Stock"] + df["Sold Stock"]
df["Target"] = (df["Available Stock"] < df["Sold Stock"]).astype(int)

# Encode categorical variables
label_cols = ["Branch Name", "Category", "Gender", "Size", "Brand"]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Sidebar filter for Dashboard
st.sidebar.header("🔍 Filter Options")
filters = {}
for col in label_cols:
    options = ["All"] + sorted(list(label_encoders[col].classes_))
    selected = st.sidebar.selectbox(col, options)
    filters[col] = selected

# Apply filters
filtered_df = df.copy()
for col in label_cols:
    if filters[col] != "All":
        encoded_val = label_encoders[col].transform([filters[col]])[0]
        filtered_df = filtered_df[filtered_df[col] == encoded_val]

# Layout & tabs
st.title("🛍️ Max Showroom Inventory Dashboard & 2025 Predictor")
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predict 2025 Restocking"])

# Dashboard tab (2023–2024)
with tab1:
    st.subheader("📦 Inventory Overview (2023–2024)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", len(filtered_df))
    col2.metric("Available Stock", int(filtered_df["Available Stock"].sum()))
    col3.metric("Sold Stock", int(filtered_df["Sold Stock"].sum()))
    col4.metric("Avg. Price", f"₹{filtered_df['Price'].mean():.2f}")

    if not filtered_df.empty:
        chart_data = filtered_df.copy()
        chart_data["Brand Name"] = label_encoders["Brand"].inverse_transform(chart_data["Brand"])
        fig = px.bar(chart_data.groupby("Brand Name")["Sold Stock"].sum().reset_index(),
                     x="Brand Name", y="Sold Stock", color="Brand Name", title="Sold Stock by Brand")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display. Try changing filters.")

# Prediction tab (2025)
with tab2:
    st.subheader("🔮 Predict Restocking Need for 2025")

    # Train-test split
    features = ["Available Stock", "Sold Stock", "Total Items", "Price"] + label_cols
    X = df[features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Form inputs
    with st.form("predict_form"):
        st.markdown("### 📄 Enter Item Details")
        branch = st.selectbox("Branch Name", label_encoders["Branch Name"].classes_)
        category = st.selectbox("Category", label_encoders["Category"].classes_)
        gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
        size = st.selectbox("Size", label_encoders["Size"].classes_)
        brand = st.selectbox("Brand", label_encoders["Brand"].classes_)
        available_stock = st.number_input("Available Stock", min_value=0, value=0)
        sold_stock = st.number_input("Sold Stock", min_value=0, value=0)
        price = st.number_input("Price", min_value=0.0, value=0.0, format="%.2f")

        submitted = st.form_submit_button("Predict Restocking Need")

    if submitted:
        total_items = available_stock + sold_stock
        st.write(f"**Total Items**: {total_items}")

        input_dict = {
            "Available Stock": available_stock,
            "Sold Stock": sold_stock,
            "Total Items": total_items,
            "Price": price,
            "Branch Name": label_encoders["Branch Name"].transform([branch])[0],
            "Category": label_encoders["Category"].transform([category])[0],
            "Gender": label_encoders["Gender"].transform([gender])[0],
            "Size": label_encoders["Size"].transform([size])[0],
            "Brand": label_encoders["Brand"].transform([brand])[0],
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df[features])[0]

        st.markdown(f"### 🧾 Prediction Result for {brand} - {category} ({size}, {gender}) at {branch}:")
        if prediction == 1:
            st.success("✅ Restocking **Needed** for 2025")
        else:
            st.info("🟢 Restocking **Not Needed** for 2025")
