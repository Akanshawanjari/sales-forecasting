
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Sales Forecasting using Multiple Linear Regression")

uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("CSV loaded successfully!")

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])

    # Create features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Features and target
    X = df[['Year', 'Month', 'Day']]
    y = df['Sales']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Model Performance")

    r2 = model.score(X, y)
    st.write("R² Score:", r2)

    # Future date input
    st.subheader("Future Sales Prediction")

    future_date = st.date_input("Select Future Date")

    future_data = pd.DataFrame({
        'Year': [future_date.year],
        'Month': [future_date.month],
        'Day': [future_date.day]
    })

    prediction = model.predict(future_data)[0]

    st.success(f"Predicted Sales: {prediction:.2f}")
