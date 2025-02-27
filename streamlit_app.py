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
EPOCHS = 500  # Default epochs, user can adjust in UI
PHYSICS_LOSS_WEIGHT = 0.1
NUM_LAGGED_FEATURES = 12

# -------------------- Physics-Informed Loss, Attention Layer, Custom Loss, PINNModel (No Changes from provided code) --------------------
def water_balance_loss(y_true, y_pred, inputs):
    pcp, temp_max, temp_min = inputs[:, 0, 0], inputs[:, 0, 1], inputs[:, 0, 2]
    et = 0.0023 * (temp_max - temp_min) * (temp_max + temp_min)
    predicted_Q = y_pred
    balance_term = pcp - (et + predicted_Q)
    return tf.reduce_mean(tf.square(balance_term))

class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

def custom_loss(inputs, y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    weights = tf.where(y_true > 0.5, 10.0, 1.0)
    weighted_mse_loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    physics_loss = water_balance_loss(y_true, y_pred, inputs)
    return weighted_mse_loss + PHYSICS_LOSS_WEIGHT * physics_loss


class PINNModel(tf.keras.Model):
    def train_step(self, data):
        X, y = data
        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss = custom_loss(X, y, y_pred)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {"loss": loss}
    def call(self, inputs):
        return super().call(inputs)


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
    dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)'] # HydroMet Features
    features = [col for col in df.columns if col != target and col in dynamic_feature_cols] # Initial HydroMet features

    # Add Lag Features for Discharge and HydroMet Features
    lagged_discharge_cols = [f'Lag_Discharge_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]
    lagged_weather_cols = [f'Lag_Rainfall_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)] + \
                          [f'Lag_TempMax_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)] + \
                          [f'Lag_TempMin_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]
    seasonality_cols = ['Month_sin', 'Month_cos'] # Seasonality Features

    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_Discharge_{lag}'] = df[target].shift(lag).fillna(0)
        if 'Rainfall (mm)' in df.columns:
            df[f'Lag_Rainfall_{lag}'] = df['Rainfall (mm)'].shift(lag).fillna(0)
        if 'Maximum temperature (°C)' in df.columns:
            df[f'Lag_TempMax_{lag}'] = df['Maximum temperature (°C)'].shift(lag).fillna(0)
        if 'Minimum temperature (°C)' in df.columns:
            df[f'Lag_TempMin_{lag}'] = df['Minimum temperature (°C)'].shift(lag).fillna(0)


    all_feature_cols = dynamic_feature_cols + lagged_discharge_cols + lagged_weather_cols + seasonality_cols
    features_for_model = [col for col in df.columns if col in all_feature_cols] # Ensure only available columns are used


    # Scaling
    scaler_dynamic = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_dynamic = df[features_for_model].values # Use selected features
    y_values = df[target].values

    X_dynamic_scaled = scaler_dynamic.fit_transform(X_dynamic)
    y_scaled = scaler_y.fit_transform(y_values.reshape(-1, 1))

    X_dynamic_scaled = X_dynamic_scaled.reshape((X_dynamic_scaled.shape[0], 1, X_dynamic_scaled.shape[1])) # Reshape for GRU


    # Train-Test Split (User-defined)
    train_split = st.slider("🎯 Select Training Data Percentage", 50, 90, 80) / 100
    X_train_dynamic, X_test_dynamic, y_train, y_test = train_test_split(X_dynamic_scaled, y_scaled, train_size=train_split, shuffle=False)
    st.write(f"📌 Train Data: {len(X_train_dynamic)}, Test Data: {len(X_test_dynamic)}")

    # User-defined epochs
    epochs = st.number_input("⏳ Set Number of Epochs", min_value=1, max_value=1000, value=EPOCHS, step=50)


    if st.button("🚀 Train Model"):
        start_time = time.time()

        # Define PINN-GRU Model
        inputs_PINN = tf.keras.Input(shape=(1, X_train_dynamic.shape[2]))
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(GRU_UNITS, return_sequences=True))(inputs_PINN)
        x = Attention()(x)
        x = tf.keras.layers.Dense(DENSE_UNITS_1, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Dense(DENSE_UNITS_2, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
        x = tf.keras.layers.Dense(DENSE_UNITS_3, activation='relu')(x)
        output_PINN = tf.keras.layers.Dense(1)(x)
        model = PINNModel(inputs_PINN, output_PINN)


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mae', run_eagerly=True) # Keep run_eagerly=True for custom loss
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, min_delta=0.001, restore_best_weights=True)


        # Train the model
        history = model.fit(X_train_dynamic, y_train, epochs=epochs, batch_size=BATCH_SIZE, validation_data=(X_test_dynamic, y_test), verbose=1,
                                          callbacks=[lr_scheduler, early_stopping])
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
        model = tf.keras.models.load_model(model_path, custom_objects={'Attention': Attention, 'PINNModel': PINNModel, 'custom_loss': custom_loss}) # custom_objects for custom layers/loss
        y_pred = model.predict(X_test_dynamic)


        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

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
        ax.plot(y_pred, label="Predicted (PINN-GRU)", color="orange") # Model name in label
        ax.set_title("📈 Actual vs. Predicted Streamflow (PINN-GRU)") # Model name in title
        ax.set_xlabel("Time")
        ax.set_ylabel("Streamflow (m³/s)")
        ax.legend()
        st.pyplot(fig)

        # Save Predictions
        results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted (PINN-GRU)": y_pred.flatten()}) # Model name in column header
        results_df.to_csv("streamflow_predictions_pinn_gru.csv", index=False) # Model name in filename
        st.download_button("📥 Download Predictions", "streamflow_predictions_pinn_gru.csv", "text/csv")
