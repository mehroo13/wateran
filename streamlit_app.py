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
from scipy.stats import pearsonr

# -------------------- Hyperparameters (PINN-GRU) --------------------
GRU_UNITS = 128
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 512
DENSE_UNITS_3 = 256
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 1000  # Default epochs, user can adjust in UI
PHYSICS_LOSS_WEIGHT = 0.1
NUM_LAGGED_FEATURES = 12
EPOCH_RANGE = list(range(1, 1001))  # Epoch range for slider
CATCHMENT_AREA_M2 = 1e6  # Placeholder: 1 km² = 1,000,000 m²; replace with your actual catchment area

# -------------------- Physics-Informed Loss, Attention Layer, Custom Loss, PINNModel --------------------
def water_balance_loss(y_true, y_pred, inputs, x_mins, x_maxs, y_min, y_max):
    """Calculate physics-based water balance loss with unscaled values."""
    # Extract scaled features
    pcp_scaled = inputs[:, 0, 0]  # Rainfall (mm), scaled
    temp_max_scaled = inputs[:, 0, 1]  # Max temperature (°C), scaled
    temp_min_scaled = inputs[:, 0, 2]  # Min temperature (°C), scaled
    predicted_Q_scaled = tf.squeeze(y_pred, axis=-1)  # Predicted discharge (m³/s), scaled

    # Unscale to original units
    pcp = pcp_scaled * (x_maxs[0] - x_mins[0]) + x_mins[0]
    temp_max = temp_max_scaled * (x_maxs[1] - x_mins[1]) + x_mins[1]
    temp_min = temp_min_scaled * (x_maxs[2] - x_mins[2]) + x_mins[2]
    predicted_Q = predicted_Q_scaled * (y_max - y_min) + y_min

    # Compute evapotranspiration, ensuring non-negative
    et = tf.maximum(
        0.0,
        0.0023 * (temp_max - temp_min) * (temp_max + temp_min)
    )

    # Convert predicted_Q from m³/s to mm/day
    conversion_factor = (86400 * 1000) / CATCHMENT_AREA_M2  # s/day * mm/m
    predicted_Q_mm_day = predicted_Q * conversion_factor

    # Compute balance term
    balance_term = pcp - (et + predicted_Q_mm_day)
    return tf.reduce_mean(tf.square(balance_term))

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def custom_loss(inputs, x_mins, x_maxs, y_min, y_max):
    """Custom loss combining MSE and physics-informed loss."""
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        weights = tf.where(y_true > 0.5, 10.0, 1.0)
        weighted_mse_loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        physics_loss = water_balance_loss(y_true, y_pred, inputs, x_mins, x_maxs, y_min, y_max)
        return weighted_mse_loss + PHYSICS_LOSS_WEIGHT * physics_loss
    return loss

class PINNModel(tf.keras.Model):
    def __init__(self, input_shape, gru_units=GRU_UNITS, dense_units_1=DENSE_UNITS_1, dense_units_2=DENSE_UNITS_2, dense_units_3=DENSE_UNITS_3, dropout_rate=DROPOUT_RATE, **kwargs):
        super(PINNModel, self).__init__(**kwargs)
        self.gru_units = gru_units
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dense_units_3 = dense_units_3
        self.dropout_rate = dropout_rate
        self.input_shape_arg = tuple(input_shape)

        # Define layers without input_shape in GRU
        self.bidirectional_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(gru_units, return_sequences=True)
        )
        self.attention = Attention()
        self.dense1 = tf.keras.layers.Dense(dense_units_1, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(dense_units_2, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense3 = tf.keras.layers.Dense(dense_units_3, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.bidirectional_gru(inputs)
        x = self.attention(x)
        x = self.dense1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        output = self.output_layer(x)
        return output  # Single output

    def get_config(self):
        config = super().get_config()
        config.update({
            'gru_units': self.gru_units,
            'dense_units_1': self.dense_units_1,
            'dense_units_2': self.dense_units_2,
            'dense_units_3': self.dense_units_3,
            'dropout_rate': self.dropout_rate,
            'input_shape': list(self.input_shape_arg)
        })
        return config

    @classmethod
    def from_config(cls, config):
        input_shape_list = config.pop('input_shape')
        input_shape = tuple(input_shape_list)
        return cls(input_shape=input_shape, **config)

# Streamlit App Title
st.title("🌊 Streamflow Prediction Web App (PINN-GRU)")

# Upload Dataset
uploaded_file = st.file_uploader("🗂 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Display dataset preview
    st.write("📊 Preview of Dataset:", df.head())

    # Handle Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df.drop(columns=['Date'], inplace=True)

    # Handle missing values - Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Select Features & Target
    target = 'Discharge (m³/S)'
    dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)']  # HydroMet Features
    features = [col for col in df.columns if col != target and col in dynamic_feature_cols]  # Initial HydroMet features

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
    scaler_dynamic = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_dynamic = df[features_for_model].values
    y_values = df[target].values

    X_dynamic_scaled = scaler_dynamic.fit_transform(X_dynamic)
    y_scaled = scaler_y.fit_transform(y_values.reshape(-1, 1))

    # Extract scaler parameters for unscaling in loss function
    x_mins = scaler_dynamic.data_min_
    x_maxs = scaler_dynamic.data_max_
    y_min = scaler_y.data_min_[0]
    y_max = scaler_y.data_max_[0]

    X_dynamic_scaled = X_dynamic_scaled.reshape((X_dynamic_scaled.shape[0], 1, X_dynamic_scaled.shape[1]))

    # Train-Test Split
    train_split = st.slider("🎯 Select Training Data Percentage", 50, 90, 80) / 100
    X_train_dynamic, X_test_dynamic, y_train, y_test = train_test_split(X_dynamic_scaled, y_scaled, train_size=train_split, shuffle=False)
    st.write(f"📌 Train Data: {len(X_train_dynamic)}, Test Data: {len(X_test_dynamic)}")

    # User-defined epochs
    epochs = st.select_slider("⏳ Set Number of Epochs:", options=EPOCH_RANGE, value=EPOCHS)

    if st.button("🚀 Train Model"):
        start_time = time.time()

        # Define PINN-GRU Model
        input_shape = (X_train_dynamic.shape[1], X_train_dynamic.shape[2])
        model = PINNModel(input_shape=input_shape)

        # Compile with custom loss including scaler parameters
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=custom_loss(X_train_dynamic, x_mins, x_maxs, y_min, y_max),
            run_eagerly=True
        )

        # Callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, min_delta=0.001, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train_dynamic, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_test_dynamic, y_test),
            verbose=1,
            callbacks=[lr_scheduler, early_stopping]
        )
        model.save("PINN_GRU_model.keras")

        training_time = time.time() - start_time
        st.write(f"✅ PINN-GRU Model Trained in {training_time:.2f} seconds!")

    if st.button("🔍 Test Model"):
        test_start_time = time.time()
        model_path = "PINN_GRU_model.keras"

        if not os.path.exists(model_path):
            st.error(f"🚨 Error: Model file '{model_path}' not found! Please train the model first.")
            st.stop()

        # Load the trained model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'Attention': Attention,
                'PINNModel': PINNModel,
                'loss': custom_loss(X_test_dynamic, x_mins, x_maxs, y_min, y_max)
            }
        )
        y_pred = model.predict(X_test_dynamic)

        # Inverse transform predictions and actual values
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        # Compute Metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        test_time = time.time() - test_start_time

        st.write(f"📉 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        st.write(f"⏳ Testing Time: {test_time:.2f} seconds!")

        # Percentage difference calculation
        percentage_difference = np.abs((y_pred.flatten() - y_actual.flatten()) / y_actual.flatten()) * 100
        percentage_difference[np.isinf(percentage_difference)] = np.nan  # Handle division by zero

        # User-defined acceptable percentage threshold
        acceptable_percentage_error = st.slider("📉 Acceptable Percentage Error Threshold", 1, 100, 20)

        # Count predictions within threshold
        within_threshold_count = np.sum(percentage_difference[~np.isnan(percentage_difference)] <= acceptable_percentage_error)
        percentage_within_threshold = (within_threshold_count / len(y_actual)) * 100 if len(y_actual) > 0 else 0

        st.write(f"✅ Predictions within ±{acceptable_percentage_error}% error: {within_threshold_count} out of {len(y_actual)} ({percentage_within_threshold:.2f}%)")

        # Plot Predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual, label="Actual", color="blue")
        ax.plot(y_pred, label="Predicted (PINN-GRU)", color="orange")
        ax.set_title("📈 Actual vs. Predicted Streamflow (PINN-GRU)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Streamflow (m³/s)")
        ax.legend()
        st.pyplot(fig)

        # Save Predictions
        results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted (PINN-GRU)": y_pred.flatten()})
        results_df.to_csv("streamflow_predictions_pinn_gru.csv", index=False)
        st.download_button("📥 Download Predictions", "streamflow_predictions_pinn_gru.csv", "text/csv")
