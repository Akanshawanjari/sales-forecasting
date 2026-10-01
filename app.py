import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Sales Forecasting using Multiple Linear Regression")

# Load CSV file
df = pd.read_csv("Sales Forecasting Dataset.csv")

st.success("CSV loaded successfully!")

st.subheader("Dataset")
st.dataframe(df)

st.subheader("Dataset Information")
st.write(df.shape)
st.write(df.dtypes)
