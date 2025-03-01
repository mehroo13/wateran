import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Model Parameters --------------------
GRU_UNITS = 64
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SPLIT = 80  # Percentage of data used for training
NUM_LAGGED_FEATURES = 3  # Number of lag features
MODEL_WEIGHTS_PATH = os.path.join(tempfile.gettempdir(), "gru_model_weights.weights.h5")

# -------------------- NSE Function --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

# -------------------- GRU Model with Dropout --------------------
def build_gru_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(GRU_UNITS, return_sequences=False, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),  # Add dropout to prevent under/overfitting
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# -------------------- Streamlit UI --------------------
st.title("📈 Streamflow Prediction using GRU")
st.markdown("This web app predicts streamflow using a GRU-based deep learning model.")

# Initialize session state to persist results
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'train_results_df' not in st.session_state:
    st.session_state.train_results_df = None
if 'test_results_df' not in st.session_state:
    st.session_state.test_results_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None

# Upload dataset
uploaded_file = st.file_uploader("📤 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 **Dataset Preview:**", df.head())

    # Check necessary columns
    required_cols = ['Date', 'Discharge (m³/S)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain the following columns: {required_cols}")
        st.stop()

    # Convert date column and create lag features
    df['Date'] = pd.to_datetime(df['Date'])
    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_{lag}'] = df['Discharge (m³/S)'].shift(lag)
    df.dropna(inplace=True)

    # User-defined model parameters
    epochs = st.slider("🔄 Number of Epochs:", min_value=1, max_value=1500, value=DEFAULT_EPOCHS, step=10)
    batch_size = st.slider("📦 Batch Size:", min_value=8, max_value=128, value=DEFAULT_BATCH_SIZE, step=8)
    train_split = st.slider("📊 Training Data Percentage:", min_value=50, max_value=90, value=DEFAULT_TRAIN_SPLIT) / 100

    # Split data first, then scale separately to avoid leakage
    train_size = int(len(df) * train_split)
    train_df, test_df = df[:train_size], df[train_size:]

    # Scale data separately for train and test
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[['Discharge (m³/S)'] + [f'Lag_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]])
    test_scaled = scaler.transform(test_df[['Discharge (m³/S)'] + [f'Lag_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]])

    # Train-test split
    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]

    # Reshape for GRU (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Debugging shapes
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Train button
    if st.button("🚀 Train Model"):
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        st.write(f"Training model with input shape: {(X_train.shape[1], X_train.shape[2])}")
        try:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
            model.save_weights(MODEL_WEIGHTS_PATH)
            st.success(f"✅ Model trained and weights saved to {MODEL_WEIGHTS_PATH}!")
        except PermissionError:
            st.error("🚨 Permission denied when saving weights. Check the directory permissions.")
        except Exception as e:
            st.error(f W"🚨 Model training or saving failed: {str(e)}")

    # Test button
    if st.button("🔍 Test Model"):
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            st.error("❌ No trained model found! Please train the model first.")
            st.stop()

        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        try:
            model.load_weights(MODEL_WEIGHTS_PATH)
            st.success("✅ Model weights loaded successfully!")
        except Exception as e:
            st.error(f"🚨 Error loading weights: {str(e)}")
            st.stop()
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Inverse transform predictions
        y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
        y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
        y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
        y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

        # Compute Metrics
        metrics = {
            "Training RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
            "Testing RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
            "Training R²": r2_score(y_train_actual, y_train_pred),
            "Testing R²": r2_score(y_test_actual, y_test_pred),
            "Training NSE": nse(y_train_actual, y_train_pred),
            "Testing NSE": nse(y_test_actual, y_test_pred)
        }

        # Check if test performance exceeds training
        if metrics["Testing RMSE"] < metrics["Training RMSE"] or metrics["Testing R²"] > metrics["Training R²"]:
            st.warning("⚠️ Test performance exceeds training performance, which is unusual. Check for data leakage or insufficient model complexity.")

        # Store results in session state
        st.session_state.metrics = metrics
        st.session_state.train_results_df = pd.DataFrame({
            "Actual_Train": y_train_actual, 
            "Predicted_Train": y_train_pred
        })
        st.session_state.test_results_df = pd.DataFrame({
            "Actual_Test": y_test_actual, 
            "Predicted_Test": y_test_pred
        })

        # Plot Predictions vs Actual Data
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(y_train_actual, label="Actual", color="blue")
        ax[0].plot(y_train_pred, label="Predicted", color="orange")
        ax[0].set_title("📈 Training Data: Actual vs. Predicted")
        ax[0].legend()
        
        ax[1].plot(y_test_actual, label="Actual", color="blue")
        ax[1].plot(y_test_pred, label="Predicted", color="orange")
        ax[1].set_title("📈 Testing Data: Actual vs. Predicted")
        ax[1].legend()

        plt.tight_layout()
        st.session_state.fig = fig

    # Display persisted results if available
    if st.session_state.metrics:
        st.json(st.session_state.metrics)
    
    if st.session_state.fig:
        st.pyplot(st.session_state.fig)

    if st.session_state.train_results_df is not None:
        train_csv = st.session_state.train_results_df.to_csv(index=False)
        st.download_button("📥 Download Training Predictions", train_csv, "train_predictions.csv", "text/csv")

    if st.session_state.test_results_df is not None:
        test_csv = st.session_state.test_results_df.to_csv(index=False)
        st.download_button("📥 Download Testing Predictions", test_csv, "test_predictions.csv", "text/csv")
