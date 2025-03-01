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
EPOCHS = 1000  # Default epochs, adjustable in UI
NUM_LAGGED_FEATURES = 12
EPOCH_RANGE = list(range(1, 1001))  # Epoch range for slider

# -------------------- GRU Model Definition --------------------
class GRUModel(tf.keras.Model):
    def __init__(self, input_shape, gru_units=GRU_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE):
        super(GRUModel, self).__init__()
        self.input_shape_arg = input_shape
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Define layers
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(dense_units, activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)

    def get_config(self):
        """Serialize the model’s configuration."""
        config = super(GRUModel, self).get_config()
        config.update({
            'input_shape': self.input_shape_arg,  # Save as tuple/list
            'gru_units': self.gru_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild the model from its configuration."""
        input_shape = config.pop('input_shape')  # Extract input_shape
        return cls(input_shape=input_shape, **config)

# Streamlit App Title
st.title("🌊 Streamflow Prediction Web App (GRU)")

# Upload Dataset
uploaded_file = st.file_uploader("🗂 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Display dataset preview
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
    lagged_weather_cols = [f'Lag_Rainfall_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)] + \
                          [f'Lag_TempMax_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)] + \
                          [f'Lag_TempMin_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]
    seasonality_cols = ['Month_sin', 'Month_cos']

    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_Discharge_{lag}'] = df[target].shift(lag).fillna(0)
        if 'Rainfall (mm)' in df.columns:
            df[f'Lag_Rainfall_{lag}'] = df['Rainfall (mm)'].shift(lag).fillna(0)
        if 'Maximum temperature (°C)' in df.columns:
            df[f'Lag_TempMax_{lag}'] = df['Maximum temperature (°C)'].shift(lag).fillna(0)
        if 'Minimum temperature (°C)' in df.columns:
            df[f'Lag_TempMin_{lag}'] = df['Minimum temperature (°C)'].shift(lag).fillna(0)

    all_feature_cols = dynamic_feature_cols + lagged_discharge_cols + lagged_weather_cols + seasonality_cols
    features_for_model = [col for col in df.columns if col in all_feature_cols]

    # Scaling
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    X_features = df[features_for_model].values
    y_values = df[target].values

    X_scaled = scaler_features.fit_transform(X_features)
    y_scaled = scaler_target.fit_transform(y_values.reshape(-1, 1))

    # Reshape for GRU (samples, timesteps, features)
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
        input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
        model = GRUModel(input_shape=input_shape)

        # Compile with MSE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse'
        )

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
        model.save("gru_model.keras")

        training_time = time.time() - start_time
        st.write(f"✅ GRU Model Trained in {training_time:.2f} seconds!")

    if st.button("🔍 Test GRU Model"):
        test_start_time = time.time()
        model_path = "gru_model.keras"

        if not os.path.exists(model_path):
            st.error(f"🚨 Error: Model file '{model_path}' not found! Please train the model first.")
        else:
            # Load the trained model with custom objects
            model = tf.keras.models.load_model(model_path, custom_objects={'GRUModel': GRUModel})
            y_pred = model.predict(X_test)

            # Inverse transform predictions and actual values
            y_pred = scaler_target.inverse_transform(y_pred.reshape(-1, 1))
            y_actual = scaler_target.inverse_transform(y_test.reshape(-1, 1))

            # Compute Metrics
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            test_time = time.time() - test_start_time

            st.write(f"📉 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            st.write(f"⏳ Testing Time: {test_time:.2f} seconds!")

            # Percentage difference
            percentage_diff = np.abs((y_pred.flatten() - y_actual.flatten()) / y_actual.flatten()) * 100
            percentage_diff[np.isinf(percentage_diff)] = np.nan  # Handle division by zero

            # Acceptable error threshold
            acceptable_error = st.slider("📉 Acceptable Error Threshold (%)", 1, 100, 20)
            within_threshold = np.sum(percentage_diff[~np.isnan(percentage_diff)] <= acceptable_error)
            percentage_within = (within_threshold / len(y_actual)) * 100 if len(y_actual) > 0 else 0

            st.write(f"✅ Predictions within ±{acceptable_error}%: {within_threshold} out of {len(y_actual)} ({percentage_within:.2f}%)")

            # Plot Predictions
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_actual, label="Actual", color="blue")
            ax.plot(y_pred, label="Predicted (GRU)", color="orange")
            ax.set_title("📈 Actual vs. Predicted Streamflow (GRU)")
            ax.set_xlabel("Time")
            ax.set_ylabel("Streamflow (m³/s)")
            ax.legend()
            st.pyplot(fig)

            # Save Predictions
            results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted (GRU)": y_pred.flatten()})
            results_df.to_csv("streamflow_predictions_gru.csv", index=False)
            st.download_button("📥 Download Predictions", "streamflow_predictions_gru.csv", "text/csv")
