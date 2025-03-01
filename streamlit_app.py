import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- Hyperparameters --------------------
GRU_UNITS = 128
DENSE_UNITS = 64
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 1000
NUM_LAGGED_FEATURES = 12
EPOCH_RANGE = list(range(1, 1001))

# -------------------- GRU Model Definition --------------------
class GRUModel(tf.keras.Model):
    def __init__(self, input_shape, gru_units=GRU_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE):
        super(GRUModel, self).__init__()
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(dense_units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)

# Streamlit App Title
st.title("🌊 Streamflow Prediction Web App (GRU)")

# Upload Dataset
uploaded_file = st.file_uploader("🗂 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 Dataset Preview:", df.head())

    # Handle Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df.drop(columns=['Date'], inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Select Features and Target
    target = 'Discharge (m³/S)'
    dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)']
    features = [col for col in df.columns if col != target and col in dynamic_feature_cols]

    # Add Lag Features
    lagged_discharge_cols = [f'Lag_Discharge_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]
    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_Discharge_{lag}'] = df[target].shift(lag).fillna(0)
        if 'Rainfall (mm)' in df.columns:
            df[f'Lag_Rainfall_{lag}'] = df['Rainfall (mm)'].shift(lag).fillna(0)
        if 'Maximum temperature (°C)' in df.columns:
            df[f'Lag_TempMax_{lag}'] = df['Maximum temperature (°C)'].shift(lag).fillna(0)
        if 'Minimum temperature (°C)' in df.columns:
            df[f'Lag_TempMin_{lag}'] = df['Minimum temperature (°C)'].shift(lag).fillna(0)

    # Scaling
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    X_features = df[lagged_discharge_cols + features].values
    y_values = df[target].values

    X_scaled = scaler_features.fit_transform(X_features)
    y_scaled = scaler_target.fit_transform(y_values.reshape(-1, 1))
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Train-Test Split
    train_split = st.slider("🎯 Training Data Percentage", 50, 90, 80) / 100
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, train_size=train_split, shuffle=False)
    st.write(f"📌 Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    # User-defined epochs
    epochs = st.select_slider("⏳ Number of Epochs:", options=EPOCH_RANGE, value=EPOCHS)

    if st.button("🚀 Train GRU Model"):
        start_time = time.time()

        # Define GRU Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = GRUModel(input_shape=input_shape)

        # Compile with MSE loss
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')

        # Callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, min_delta=0.001, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[lr_scheduler, early_stopping]
        )

        # Save the model in TensorFlow's SavedModel format (instead of HDF5)
        model.save("gru_model")  
        training_time = time.time() - start_time
        st.write(f"✅ GRU Model Trained in {training_time:.2f} seconds!")

    if st.button("🔍 Test GRU Model"):
        test_start_time = time.time()

        # Ensure model exists
        model_path = "gru_model"
        if not os.path.exists(model_path):
            st.error(f"🚨 Error: Model directory '{model_path}' not found! Please train the model first.")
            st.stop()

        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(X_test)

        # Inverse transform predictions
        y_pred = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
        y_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

        # Compute Metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        test_time = time.time() - test_start_time

        st.write(f"📉 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        st.write(f"⏳ Testing Time: {test_time:.2f} seconds!")

        # Plot Predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual, label="Actual", color="blue")
        ax.plot(y_pred, label="Predicted (GRU)", color="orange")
        ax.set_title("📈 Actual vs. Predicted Streamflow (GRU)")
        ax.legend()
        st.pyplot(fig)

        # Save Predictions
        results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted (GRU)": y_pred.flatten()})
        st.download_button("📥 Download Predictions", results_df.to_csv(index=False), "streamflow_predictions.csv", "text/csv")
