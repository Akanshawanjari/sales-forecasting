
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Sales Forecasting using Multiple Linear Regression")

# Load dataset automatically
df = pd.read_csv("Sales Forecasting Dataset.csv")

st.success("Sales dataset loaded successfully!")

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

# Dataset
st.subheader("Dataset")
st.dataframe(df)

# Model Performance
st.subheader("Model Performance")

r2 = model.score(X, y)
st.write("R² Score:", r2)
st.write("R² Score (%):", round(r2 * 100, 2), "%")

# Future Sales Prediction
st.subheader("Future Sales Prediction")

future_date = st.date_input("Select Future Date")

future_data = pd.DataFrame({
    'Year': [future_date.year],
    'Month': [future_date.month],
    'Day': [future_date.day]
})

prediction = model.predict(future_data)[0]

st.success(f"Predicted Sales: {prediction:.2f}")
