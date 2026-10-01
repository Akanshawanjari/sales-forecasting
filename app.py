import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting using Multiple Linear Regression")

st.write(
    "Analyze historical sales data and predict future sales trends "
    "using Multiple Linear Regression."
)

# ---------------------------------------------------
# LOAD CSV
# ---------------------------------------------------

file_name = "Sales Forecasting Dataset.csv"

if os.path.exists(file_name):
    # Automatically load CSV from GitHub
    df = pd.read_csv(file_name)
    st.success("✅ Sales Forecasting Dataset loaded successfully!")

else:
    # If CSV is not found, allow user to upload it
    st.warning("⚠️ CSV file not found in the repository.")
    uploaded_file = st.file_uploader(
        "Upload Sales Forecasting CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Please upload the CSV file to continue.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV uploaded successfully!")


# ---------------------------------------------------
# DATASET
# ---------------------------------------------------

st.subheader("📋 Dataset")

st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------

st.subheader("🔍 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])

with col2:
    st.write("Dataset Shape:", df.shape)

st.subheader("📌 Data Types")
st.write(df.dtypes)


# ---------------------------------------------------
# DATE CONVERSION
# ---------------------------------------------------

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid dates
    df = df.dropna(subset=["Date"])

    # Create date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    st.success("✅ Date converted successfully!")


# ---------------------------------------------------
# FIND SALES COLUMN
# ---------------------------------------------------

sales_column = None

possible_sales_columns = [
    "Sales",
    "sales",
    "Total Sales",
    "Total_Sales",
    "Revenue",
    "revenue"
]

for column in possible_sales_columns:
    if column in df.columns:
        sales_column = column
        break


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

if sales_column is not None:

    st.subheader("🤖 Multiple Linear Regression Model")

    # Convert sales column to numeric
    df[sales_column] = pd.to_numeric(
        df[sales_column],
        errors="coerce"
    )

    df = df.dropna(subset=[sales_column])

    # Features
    features = ["Year", "Month", "Day"]

    X = df[features]
    y = df[sales_column]

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X, y)

    # Predictions
    df["Predicted Sales"] = model.predict(X)

    st.success("✅ Model trained successfully!")

    # ---------------------------------------------------
    # MODEL RESULTS
    # ---------------------------------------------------

    st.subheader("📈 Prediction Results")

    st.dataframe(
        df[["Date", sales_column, "Predicted Sales"]],
        use_container_width=True
    )

    # ---------------------------------------------------
    # FUTURE PREDICTION
    # ---------------------------------------------------

    st.subheader("🔮 Future Sales Prediction")

    future_date = st.date_input(
        "Select a future date"
    )

    future_date = pd.to_datetime(future_date)

    future_data = pd.DataFrame({
        "Year": [future_date.year],
        "Month": [future_date.month],
        "Day": [future_date.day]
    })

    prediction = model.predict(future_data)

    st.metric(
        "Predicted Sales",
        f"{prediction[0]:,.2f}"
    )

    # ---------------------------------------------------
    # COEFFICIENTS
    # ---------------------------------------------------

    st.subheader("📊 Model Coefficients")

    coefficient_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    st.dataframe(
        coefficient_df,
        use_container_width=True
    )

else:

    st.error(
        "❌ Sales column not found in your CSV file."
    )

    st.write("Your CSV columns are:")
    st.write(list(df.columns))

    st.info(
        "Please make sure your dataset has a column named "
        "'Sales' or 'Revenue'."
    )
