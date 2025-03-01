import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------- Model Parameters --------------------
GRU_UNITS = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 50  # Reduced for simplicity
NUM_LAGGED_FEATURES = 3  # Reduced for simplicity
MODEL_WEIGHTS_PATH = "gru_weights.tf"

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
st.title("📈 Simple GRU Streamflow Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:", df.head())

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

    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Discharge (m³/S)'] + [f'Lag_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]])

    # Train-test split
    X, y = scaled_data[:, 1:], scaled_data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Reshape for GRU (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Train button
    if st.button("🚀 Train Model"):
        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        model.save_weights(MODEL_WEIGHTS_PATH)  # Save model weights
        st.success("✅ Model trained and saved!")

    # Test button
    if st.button("🔍 Test Model"):
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            st.error("❌ No trained model found! Please train the model first.")
            st.stop()

        model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        model.load_weights(MODEL_WEIGHTS_PATH)  # Load model weights
        y_pred = model.predict(X_test)

        # Inverse transform predictions
        y_pred = scaler.inverse_transform(np.hstack([y_pred, X_test[:, 0, :]]))[:, 0]
        y_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        st.write(f"📉 RMSE: {rmse:.4f}")

