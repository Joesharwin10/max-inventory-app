import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Max Inventory 2025 Prediction", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Max 2023-2024!.csv")

df = load_data()

# Add Target
df["Target"] = (df["Available Stock"] < df["Sold Stock"]).astype(int)

# Label Encoding
label_cols = ["Branch Name", "Category", "Gender", "Size", "Brand", "Season", "Season Month"]
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Tabs
st.title("🛍️ Max Showroom Inventory Dashboard & 2025 Predictor")
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Predict 2025 Restocking"])

# --- Tab 1: Dashboard ---
with tab1:
    st.sidebar.header("🔍 Filter Options")
    filters = {}
    for col in ["Branch Name", "Category", "Gender", "Size", "Brand"]:
        options = ["All"] + list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(col, options)
        filters[col] = selected

    filtered_df = df.copy()
    for col in ["Branch Name", "Category", "Gender", "Size", "Brand"]:
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
        fig = px.bar(chart_df.groupby("Brand Name")["Sold Stock"].sum().reset_index(),
                     x="Brand Name", y="Sold Stock", color="Brand Name", title="Sold Stock by Brand")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available. Try changing filters.")

# --- Tab 2: Prediction ---
with tab2:
    st.subheader("🔮 Predict Restocking for 2025")

    # Train model
    X = df[["Available Stock", "Sold Stock"] + label_cols]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Form
    with st.form("predict_form"):
        st.markdown("### 📄 Enter Item Details")
        branch = st.selectbox("Branch Name", label_encoders["Branch Name"].classes_)
        category = st.selectbox("Category", label_encoders["Category"].classes_)
        gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
        size = st.selectbox("Size", label_encoders["Size"].classes_)
        brand = st.selectbox("Brand", label_encoders["Brand"].classes_)
        season = st.selectbox("Season", label_encoders["Season"].classes_)
        season_month = st.selectbox("Season Month", label_encoders["Season Month"].classes_)
        available_stock = st.number_input("Available Stock", min_value=0, value=0)
        sold_stock = st.number_input("Sold Stock", min_value=0, value=0)
        submit = st.form_submit_button("Predict")

    # Show full inventory data with Timestamp
    display_df = df.copy()
    for col in label_cols:
        display_df[col] = label_encoders[col].inverse_transform(display_df[col])

    st.markdown("#### 📂 Inventory Past Data (with Timestamp)")
    st.dataframe(display_df[[
        "Timestamp", "Branch Name", "Category", "Brand", "Size", "Gender",
        "Season", "Season Month", "Available Stock", "Sold Stock"
    ]])

    if submit:
        input_row = {
            "Available Stock": available_stock,
            "Sold Stock": sold_stock,
            "Branch Name": label_encoders["Branch Name"].transform([branch])[0],
            "Category": label_encoders["Category"].transform([category])[0],
            "Gender": label_encoders["Gender"].transform([gender])[0],
            "Size": label_encoders["Size"].transform([size])[0],
            "Brand": label_encoders["Brand"].transform([brand])[0],
            "Season": label_encoders["Season"].transform([season])[0],
            "Season Month": label_encoders["Season Month"].transform([season_month])[0],
        }

        matched_df = df[
            (df["Branch Name"] == input_row["Branch Name"]) &
            (df["Category"] == input_row["Category"]) &
            (df["Gender"] == input_row["Gender"]) &
            (df["Size"] == input_row["Size"]) &
            (df["Brand"] == input_row["Brand"]) &
            (df["Season"] == input_row["Season"]) &
            (df["Season Month"] == input_row["Season Month"]) &
            (df["Available Stock"] == available_stock) &
            (df["Sold Stock"] == sold_stock)
        ]

        input_df = pd.DataFrame([input_row])
        prediction = model.predict(input_df)[0]

        st.markdown(f"### 🧾 Prediction for {category} - {brand} ({size}, {gender}) at {branch} for **{season_month}**")

        if prediction == 1:
            qty = max(int(sold_stock - available_stock), 1)
            st.success(f"⚠️ The month of (**{season_month}**) restocking is **needed** – Suggested Quantity: **{qty}** items")
        else:
            st.info(f"✅ The month of (**{season_month}**) restocking is **not needed**.")

        if not matched_df.empty:
            st.success("✅ Exact matching records found below:")
        else:
            st.warning("⚠️ No exact past records found. Table shown for manual verification.")

        matched_display_df = matched_df.copy()
        for col in label_cols:
            matched_display_df[col] = label_encoders[col].inverse_transform(matched_display_df[col])

        st.dataframe(matched_display_df[[
            "Timestamp", "Branch Name", "Category", "Brand", "Size", "Gender",
            "Season", "Season Month", "Available Stock", "Sold Stock"
        ]])
