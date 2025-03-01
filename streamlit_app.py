import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
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
MODEL_WEIGHTS_PATH = "gru_weights.weights.h5"

# -------------------- NSE Function --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

# -------------------- Simple GRU Model --------------------
def build_gru_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(GRU_UNITS, return_sequences=False, input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model

# -------------------- Streamlit UI --------------------
st.title("📈 Streamflow Prediction using GRU")

# Upload dataset
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 Dataset Preview:", df.head())

    # Ensure necessary columns exist
    required_cols = ['Date', 'Discharge (m³/S)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset must contain the following columns: {required_cols}")
        st.stop()

    # Convert date column and create lag features
    df['Date'] = pd.to_datetime(df['Date'])
    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_{lag}'] = df['Discharge (m³/S)'].shift(lag)
    df.dropna(inplace=True)

    # User input for training parameters
    epochs = st.number_input("🔄 Number of Epochs:", min_value=10, max_value=500, value=DEFAULT_EPOCHS, step=10)
    batch_size = st.number_input("📦 Batch Size:", min_value=8, max_value=128, value=DEFAULT_BATCH_SIZE, step=8)
    train_split = st.slider("📊 Training Data Percentage:", min_value=50, max_value=90, value=DEFAULT_TRAIN_SPLIT) / 100

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Discharge (m³/S)'] + [f'Lag_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]])

    # Train-test split
    train_size = int(len(scaled_data) * train_split)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]

    # Reshape for GRU (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Train button
    if st.button("🚀 Train Model"):
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        model.save_weights(MODEL_WEIGHTS_PATH)  # Save model weights
        st.success("✅ Model trained and saved!")

    # Test button
    if st.button("🔍 Test Model"):
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            st.error("❌ No trained model found! Please train the model first.")
            st.stop()

        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.load_weights(MODEL_WEIGHTS_PATH)  # Load model weights

        # Make predictions for both training and testing sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Inverse transform predictions
        y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
        y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
        y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
        y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

        # Compute Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
        train_r2 = r2_score(y_train_actual, y_train_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)
        train_nse = nse(y_train_actual, y_train_pred)
        test_nse = nse(y_test_actual, y_test_pred)

        # Display Metrics
        st.write(f"📊 **Training Performance**: RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}, NSE = {train_nse:.4f}")
        st.write(f"📊 **Testing Performance**: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, NSE = {test_nse:.4f}")

        # Plot Predictions vs Actual Data
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        
        # Training plot
        ax[0].plot(y_train_actual, label="Actual", color="blue")
        ax[0].plot(y_train_pred, label="Predicted", color="orange")
        ax[0].set_title("📈 Training Data: Actual vs. Predicted")
        ax[0].legend()

        # Testing plot
        ax[1].plot(y_test_actual, label="Actual", color="blue")
        ax[1].plot(y_test_pred, label="Predicted", color="orange")
        ax[1].set_title("📈 Testing Data: Actual vs. Predicted")
        ax[1].legend()

        st.pyplot(fig)

        # Save Predictions
        results_df = pd.DataFrame({
            "Actual_Train": np.concatenate([y_train_actual, np.full(len(y_test_actual), np.nan)]),
            "Predicted_Train": np.concatenate([y_train_pred, np.full(len(y_test_actual), np.nan)]),
            "Actual_Test": np.concatenate([np.full(len(y_train_actual), np.nan), y_test_actual]),
            "Predicted_Test": np.concatenate([np.full(len(y_train_actual), np.nan), y_test_pred])
        })
        
        csv_file = results_df.to_csv(index=False)
        st.download_button("📥 Download Predictions", csv_file, "predictions.csv", "text/csv")
