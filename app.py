import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Max Inventory 2025 Prediction", layout="wide")

# Load and cache data
@st.cache_data
def load_data():
    return pd.read_csv("Max 2023-2024!.csv")

df = load_data()

# Drop unwanted columns
columns_to_drop = ["Timestamp", "Restock Needed", "Data Split", "Target"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Label Encoding
label_cols = ["Branch Name", "Category", "Gender", "Size", "Brand", "Season", "Season Month"]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Train model
X = df[["Available Stock", "Sold Stock"] + label_cols]
y = (df["Available Stock"] < df["Sold Stock"]).astype(int)  # Dynamic target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tabs
st.title("🛍️ Max Showroom Inventory Dashboard & 2025 Prediction")
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Prediction of 2025 Restocking"])

# --- TAB 1: Dashboard ---
with tab1:
    st.sidebar.header("🔍 Filter Options")
    filters = {}
    for col in ["Branch Name", "Category", "Gender", "Size", "Brand"]:
        options = ["All"] + list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(col, options)
        filters[col] = selected

    filtered_df = df.copy()
    for col in filters:
        if filters[col] != "All":
            val = label_encoders[col].transform([filters[col]])[0]
            filtered_df = filtered_df[filtered_df[col] == val]

    st.subheader("📦 Inventory Summary (2023–2024)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", len(filtered_df))
    col2.metric("Total Stock", int(filtered_df["Available Stock"].sum()))
    col3.metric("Sold Stock", int(filtered_df["Sold Stock"].sum()))
    col4.metric("Avg Price", f"₹{filtered_df['Price'].mean():.2f}")

    if not filtered_df.empty:
        chart_df = filtered_df.copy()
        chart_df["Brand Name"] = label_encoders["Brand"].inverse_transform(chart_df["Brand"])
        brand_chart = px.bar(
            chart_df.groupby("Brand Name")["Sold Stock"].sum().reset_index(),
            x="Brand Name", y="Sold Stock", color="Brand Name", title="Sold Stock by Brand"
        )
        st.plotly_chart(brand_chart, use_container_width=True)
    else:
        st.warning("⚠️ No data available for selected filters.")

# --- TAB 2: Prediction ---
with tab2:
    st.subheader("🔮 Predicting Restocking Requirement for 2025")

    with st.form("predict_form"):
        st.markdown("### 📄 Enter Item Details")

        form_inputs = {}
        for col in label_cols:
            label = col.replace("_", " ")
            form_inputs[col] = st.selectbox(label, label_encoders[col].classes_)

        available_stock = st.number_input("Available Stock", min_value=0, value=0)
        sold_stock = st.number_input("Sold Stock", min_value=0, value=0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = {
            "Available Stock": available_stock,
            "Sold Stock": sold_stock
        }
        for col in label_cols:
            input_data[col] = label_encoders[col].transform([form_inputs[col]])[0]

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        st.markdown("### 🧾 Prediction Result")
        summary = f"**{form_inputs['Category']} - {form_inputs['Brand']} ({form_inputs['Size']}, {form_inputs['Gender']})** at **{form_inputs['Branch Name']}** for **{form_inputs['Season Month']}**"
        if prediction == 1:
            suggested_qty = max(int(sold_stock - available_stock), 1)
            st.success(f"⚠️ Restocking is **needed** for {summary} – Suggested Quantity: **{suggested_qty}** items")
        else:
            st.info(f"✅ Restocking is **not needed** for {summary}")

        # Matching Past Records
        matched_df = df.copy()
        for col in label_cols:
            matched_df = matched_df[matched_df[col] == input_data[col]]
        matched_df = matched_df[
            (matched_df["Available Stock"] == available_stock) &
            (matched_df["Sold Stock"] == sold_stock)
        ]

        # Decode matched records
        for col in label_cols:
            matched_df[col] = label_encoders[col].inverse_transform(matched_df[col])

        if not matched_df.empty:
            st.success("✅ Matching historical records found:")
        else:
            st.warning("⚠️ No exact match found. Showing possible related records.")

        st.markdown("#### 📌 Matching Records")
        st.dataframe(matched_df)

    # --- Show All Data (excluding removed columns) ---
    display_df = df.copy()
    for col in label_cols:
        display_df[col] = label_encoders[col].inverse_transform(display_df[col])
    st.markdown("#### 📂 Complete Inventory Records (2023–2024)")
    st.dataframe(display_df)

