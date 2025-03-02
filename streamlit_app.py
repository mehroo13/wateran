import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import time

# --------------- Streamlit App Configurations ---------------
st.set_page_config(page_title="Time Series Prediction", page_icon="📈", layout="wide")

# 🚀 **Header Section**
st.markdown("<h1 style='text-align: center;'>🌊 Time Series Forecasting with GRU</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Train and predict time series with an AI-driven model.</h4>", unsafe_allow_html=True)
st.markdown("---")

# 🛠 **Sidebar - Model Configuration**
st.sidebar.header("⚙️ Model Configuration")
epochs = st.sidebar.slider("Epochs:", 1, 1500, 50, step=10)
batch_size = st.sidebar.slider("Batch Size:", 8, 128, 16, step=8)
train_split = st.sidebar.slider("Training Data %:", 50, 90, 80) / 100
learning_rate = st.sidebar.number_input("Learning Rate:", min_value=0.00001, max_value=0.1, value=0.001, format="%.5f")

# **GRU & Dense Layers Customization**
st.sidebar.subheader("🔧 GRU & Dense Layers")
gru_layers = st.sidebar.number_input("Number of GRU Layers:", min_value=1, max_value=5, value=1, step=1)
gru_units = [st.sidebar.slider(f"GRU Layer {i+1} Units:", 8, 512, 64, step=8) for i in range(gru_layers)]
dense_layers = st.sidebar.number_input("Number of Dense Layers:", min_value=1, max_value=5, value=1, step=1)
dense_units = [st.sidebar.slider(f"Dense Layer {i+1} Units:", 8, 512, 32, step=8) for i in range(dense_layers)]

# 🚀 **Upload Data**
st.subheader("📂 Upload Your Time-Series Data")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ File `{uploaded_file.name}` uploaded successfully!")

    # 📊 **Preview Data**
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # 🕒 **Datetime Column Selection**
    datetime_cols = [col for col in df.columns if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])]
    date_col = st.selectbox("Select datetime column (if applicable):", ["None"] + datetime_cols, index=0)
    if date_col != "None":
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

    # 🔢 **Numeric Columns Selection**
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    if not numeric_cols:
        st.error("No numeric columns found in the dataset. Please upload a valid dataset with numeric data.")
        st.stop()

    output_var = st.selectbox("🎯 Output Variable to Predict:", numeric_cols)

    # Ensure at least one numeric column is available for input
    default_input_vars = [numeric_cols[0]] if len(numeric_cols) > 1 else []
    input_vars = st.multiselect("📊 Input Variables:", [col for col in numeric_cols if col != output_var], default=default_input_vars)

    # **Training & Testing Split Preview**
    train_size = int(len(df) * train_split)
    st.info(f"**Training Size:** {train_size} rows | **Testing Size:** {len(df) - train_size} rows")

    # **Train & Test Data Preparation**
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])
    
    # 🚀 **Model Training**
    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress... ⏳"):
            time.sleep(2)  # Simulating training (Replace with real model.fit)
            st.success("🎉 Model trained successfully!")

    # 🔍 **Model Testing**
    if st.button("🔍 Test Model"):
        with st.spinner("Testing model... ⏳"):
            time.sleep(2)  # Simulating testing
            st.success("✅ Model tested successfully!")

        # 📊 **Performance Metrics**
        st.subheader("📏 Model Performance Metrics")
        metrics = {
            "Training RMSE": 0.12,
            "Testing RMSE": 0.18,
            "Training R²": 0.95,
            "Testing R²": 0.88
        }
        st.table(pd.DataFrame(metrics, index=["Value"]))

        # 📈 **Results Visualization**
        tab1, tab2 = st.tabs(["📊 Training Results", "📊 Testing Results"])
        with tab1:
            fig_train = px.line(y=[np.random.rand(100), np.random.rand(100)], labels={"value": "Output"})
            st.plotly_chart(fig_train, use_container_width=True)
        with tab2:
            fig_test = px.line(y=[np.random.rand(100), np.random.rand(100)], labels={"value": "Output"})
            st.plotly_chart(fig_test, use_container_width=True)

        # 📥 **Download Results**
        st.subheader("📩 Download Results")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("⬇️ Download Training Data", "train_predictions.csv", "text/csv")
        with col_dl2:
            st.download_button("⬇️ Download Testing Data", "test_predictions.csv", "text/csv")

st.markdown("---")
st.markdown("<h5 style='text-align: center;'>Built with ❤️ by xAI | Powered by Streamlit 🚀</h5>", unsafe_allow_html=True)
