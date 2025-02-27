import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Streamlit App Title
st.title("🌊 Streamflow Prediction Web App")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload an Excel file", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("\U0001F4CA Preview of Dataset:", df.head())

    # Handle Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Select Features & Target
    target = 'Discharge (m³/S)'
    features = [col for col in df.columns if col != target]

    # Add Lag Features
    NUM_LAGGED_FEATURES = 12
    for lag in range(1, NUM_LAGGED_FEATURES + 1):
        df[f'Lag_Discharge_{lag}'] = df[target].shift(lag).fillna(0)

    # Scaling
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_x.fit_transform(df[features])
    y = scaler_y.fit_transform(df[target].values.reshape(-1, 1))

    # Train-Test Split (User-defined)
    train_split = st.slider("\U0001F3AF Select Training Data Percentage", 50, 90, 80) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=False)
    st.write(f"\U0001F4CC Train Data: {len(X_train)}, Test Data: {len(X_test)}")

    # Model Selection
    model_choice = st.selectbox("\U0001F4E1 Choose a Model", ["GRU", "Random Forest", "XGBoost"])

    # Allow user to set number of epochs for GRU
    if model_choice == "GRU":
        epochs = st.slider("\U0001F4BB Select Number of Epochs", 10, 500, 100)

    if st.button("\U0001F680 Train Model"):
        start_time = time.time()

        if model_choice == "GRU":
            X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            model = tf.keras.Sequential([
                tf.keras.layers.GRU(128, return_sequences=True, input_shape=(1, X_train.shape[1])),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae')
            model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
            model.save("GRU_model.h5")

        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train.ravel())
            joblib.dump(model, "RF_model.pkl")

        elif model_choice == "XGBoost":
            model = XGBRegressor(n_estimators=100)
            model.fit(X_train, y_train.ravel())
            joblib.dump(model, "XGB_model.pkl")

        training_time = time.time() - start_time
        st.write(f"✅ Model Trained in {training_time:.2f} seconds!")

    if st.button("\U0001F50D Test Model"):
        test_start_time = time.time()

        # Load the correct model
        if model_choice == "Random Forest":
            model = joblib.load("RF_model.pkl")
        elif model_choice == "XGBoost":
            model = joblib.load("XGB_model.pkl")
        elif model_choice == "GRU":
            model = tf.keras.models.load_model("GRU_model.h5")
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Predict
        y_pred = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        # Compute Metrics
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        st.write(f"📉 RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Plot Predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual, label="Actual", color="blue")
        ax.plot(y_pred, label="Predicted", color="orange")
        ax.set_title("📈 Actual vs. Predicted Streamflow")
        ax.set_xlabel("Time")
        ax.set_ylabel("Streamflow (m³/s)")
        ax.legend()
        st.pyplot(fig)

        # Save Predictions
        results_df = pd.DataFrame({"Actual": y_actual.flatten(), "Predicted": y_pred.flatten()})
        results_df.to_csv("streamflow_predictions.csv", index=False)
        st.download_button("📥 Download Predictions", "streamflow_predictions.csv", "text/csv")
