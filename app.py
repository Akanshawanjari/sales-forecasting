import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊"
)

st.title("📊 Sales Forecasting using Multiple Linear Regression")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload Sales Forecasting Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("✅ Sales dataset loaded successfully!")

    # Check columns
    if "Date" not in df.columns:
        st.error("❌ 'Date' column is missing.")
        st.write("Available columns:", list(df.columns))
        st.stop()

    if "Sales" not in df.columns:
        st.error("❌ 'Sales' column is missing.")
        st.write("Available columns:", list(df.columns))
        st.stop()

    # Convert Date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert Sales
    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    # Remove invalid data
    df = df.dropna(
        subset=["Date", "Sales"]
    )

    if len(df) < 2:
        st.error("❌ Not enough valid data for prediction.")
        st.stop()

    # Create features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Features and target
    X = df[["Year", "Month", "Day"]]
    y = df["Sales"]

    # Train Multiple Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Dataset
    st.subheader("📋 Dataset")
    st.dataframe(df, use_container_width=True)

    # Model Performance
    st.subheader("📈 Model Performance")

    r2 = model.score(X, y)

    st.write("R² Score:", round(r2, 4))
    st.write(
        "R² Score (%):",
        round(r2 * 100, 2),
        "%"
    )

    # Future Sales Prediction
    st.subheader("🔮 Future Sales Prediction")

    future_date = st.date_input(
        "Select Future Date"
    )

    future_data = pd.DataFrame({
        "Year": [future_date.year],
        "Month": [future_date.month],
        "Day": [future_date.day]
    })

    prediction = model.predict(
        future_data
    )[0]

    st.success(
        f"💰 Predicted Sales: {prediction:.2f}"
    )

else:
    st.info(
        "👆 Please upload your Sales Forecasting Dataset CSV file."
    )
