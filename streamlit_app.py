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
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 512
DENSE_UNITS_3 = 256
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 1000
PHYSICS_LOSS_WEIGHT = 0.1
NUM_LAGGED_FEATURES = 12
EPOCH_RANGE = list(range(1, 1001))

# -------------------- Custom Layers and Loss Functions --------------------

def water_balance_loss(y_true, y_pred, inputs):
    """Calculate physics-based loss enforcing water balance."""
    pcp = inputs[:, 0, 0]  # Rainfall
    temp_max = inputs[:, 0, 1]  # Max temperature
    temp_min = inputs[:, 0, 2]  # Min temperature
    et = 0.0023 * (temp_max - temp_min) * (temp_max + temp_min)  # Evapotranspiration
    predicted_Q = tf.squeeze(y_pred, axis=-1)  # Ensure y_pred is (batch_size,)
    balance_term = pcp - (et + predicted_Q)
    return tf.reduce_mean(tf.square(balance_term))

class Attention(tf.keras.layers.Layer):
    """Custom attention layer."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)

    def get_config(self):
        return super().get_config()

def custom_loss(inputs):
    """Custom loss combining MSE and physics loss."""
    def loss(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        weights = tf.where(y_true > 0.5, 10.0, 1.0)
        weighted_mse_loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
        physics_loss = water_balance_loss(y_true, y_pred, inputs)
        return weighted_mse_loss + PHYSICS_LOSS_WEIGHT * physics_loss
    return loss

class PINNModel(tf.keras.Model):
    """PINN-GRU model."""
    def __init__(self, input_shape, gru_units=GRU_UNITS, dense_units_1=DENSE_UNITS_1, dense_units_2=DENSE_UNITS_2, dense_units_3=DENSE_UNITS_3, dropout_rate=DROPOUT_RATE):
        super(PINNModel, self).__init__()
        self.input_shape_arg = input_shape
        self.bidirectional_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=True))
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
        return self.output_layer(x)  # Single output

    def get_config(self):
        return {
            'input_shape': list(self.input_shape_arg),
            'gru_units': GRU_UNITS,
            'dense_units_1': DENSE_UNITS_1,
            'dense_units_2': DENSE_UNITS_2,
            'dense_units_3': DENSE_UNITS_3,
            'dropout_rate': DROPOUT_RATE
        }

    @classmethod
    def from_config(cls, config):
        input_shape = tuple(config.pop('input_shape'))
        return cls(input_shape=input_shape, **config)

# -------------------- Streamlit App --------------------

st.title("🌊 Streamflow Prediction Web App (PINN-GRU)")

# Upload Dataset
uploaded_file = st.file_uploader("🗂 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("📊 Preview of Dataset:", df.head())

    # Preprocess Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df.drop(columns=['Date'], inplace=True)

    df.fillna(0, inplace=True)

    # Define Features and Target
    target = 'Discharge (m³/S)'
    dynamic_feature_cols = ['Rainfall (mm)', 'Maximum temperature (°C)', 'Minimum temperature (°C)']
    features = [col for col in df.columns if col != target and col in dynamic_feature_cols]

    # Add Lagged Features
    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_Discharge_{lag}'] = df[target].shift(lag).fillna(0)
        for col in dynamic_feature_cols:
            if col in df.columns:
                df[f'Lag_{col}_{lag}'] = df[col].shift(lag).fillna(0)

    all_feature_cols = dynamic_feature_cols + [f'Lag_Discharge_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)] + \
                       [f'Lag_{col}_{i}' for col in dynamic_feature_cols for i in range(1, NUM_LAGGED_FEATURES + 1) if f'Lag_{col}_{i}' in df.columns] + \
                       ['Month_sin', 'Month_cos']
    features_for_model = [col for col in df.columns if col in all_feature_cols]

    # Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(df[features_for_model].values)
    y = scaler_y.fit_transform(df[target].values.reshape(-1, 1))
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Shape: (samples, timesteps, features)

    # Train-Test Split
    train_split = st.slider("🎯 Training Data Percentage", 50, 90, 80) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=False)
    st.write(f"📌 Train Data: {len(X_train)}, Test Data: {len(X_test)}")

    # Epoch Selection
    epochs = st.select_slider("⏳ Epochs:", options=EPOCH_RANGE, value=EPOCHS)

    if st.button("🚀 Train Model"):
        start_time = time.time()
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = PINNModel(input_shape=input_shape)
        
        # Compile with custom loss
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
                      loss=custom_loss(X_train), 
                      run_eagerly=True)
        
        # Callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, min_delta=0.001, restore_best_weights=True)

        # Train
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, 
                            validation_data=(X_test, y_test), verbose=1, 
                            callbacks=[lr_scheduler, early_stopping])
        model.save("PINN_GRU_model.keras")
        
        st.write(f"✅ Model Trained in {time.time() - start_time:.2f} seconds!")

    if st.button("🔍 Test Model"):
        if not os.path.exists("PINN_GRU_model.keras"):
            st.error("🚨 Model not found! Please train the model first.")
        else:
            model = tf.keras.models.load_model("PINN_GRU_model.keras", 
                                               custom_objects={'Attention': Attention, 'PINNModel': PINNModel, 'loss': custom_loss(X_test)})
            y_pred = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred)
            y_actual = scaler_y.inverse_transform(y_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            st.write(f"📉 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_actual, label="Actual", color="blue")
            ax.plot(y_pred, label="Predicted (PINN-GRU)", color="orange")
            ax.set_title("📈 Actual vs. Predicted Streamflow")
            ax.set_xlabel("Time")
            ax.set_ylabel("Streamflow (m³/s)")
            ax.legend()
            st.pyplot(fig)

            # Save Predictions
            results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted": y_pred.flatten()})
            st.download_button("📥 Download Predictions", results_df.to_csv(index=False), "predictions.csv", "text/csv")
