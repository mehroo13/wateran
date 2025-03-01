import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration ---
st.set_page_config(
    page_title="Streamflow Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com', # Replace with your help URL
        'Report a bug': "https://www.example.com", # Replace with bug report URL
        'About': "This app predicts streamflow using a GRU model."
    }
)

# -------------------- Model Parameters --------------------
GRU_UNITS = 64
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100 # Increased default epochs for potentially better training
DEFAULT_BATCH_SIZE = 32 # Increased default batch size for potentially faster training
DEFAULT_TRAIN_SPLIT = 80
NUM_LAGGED_FEATURES = 3
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
st.markdown("Welcome to the Streamflow Prediction App! Upload your streamflow data, train a GRU model, and evaluate its performance.")

# --- File Upload and Data Loading ---
st.header("Step 1: Upload Dataset")
uploaded_file = st.file_uploader(
    "Upload an Excel or CSV file containing time series streamflow data",
    type=["xlsx", "csv"],
    help="""
        Upload a file with 'Date' and 'Discharge (m³/S)' columns.
        - **Excel (.xlsx):** Ensure data is in a single sheet.
        - **CSV (.csv):**  Use comma as a delimiter.
    """,
    label_visibility="visible" ,
    accept_multiple_files=False,
    key="file_uploader"
)

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    try:
        if file_type == 'xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload an Excel (.xlsx) or CSV (.csv) file.")
            st.stop()

        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
        st.write("📊 Dataset Preview:", df.head())

        # --- Data Validation ---
        required_cols = ['Date', 'Discharge (m³/S)']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dataset must contain the following columns: {required_cols}")
            st.stop()

        # --- Data Preprocessing ---
        with st.spinner("Processing data..."):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except ValueError:
                st.error("Error converting 'Date' column to datetime. Please ensure 'Date' column is in a recognizable date format.")
                st.stop()
            try:
                df['Discharge (m³/S)'] = pd.to_numeric(df['Discharge (m³/S)'], errors='raise') # Raise error for non-numeric values
            except ValueError:
                st.error("Error converting 'Discharge (m³/S)' column to numeric. Please ensure this column contains only numeric values.")
                st.stop()

            if df.empty:
                st.error("Uploaded file is empty after processing. Please check your file.")
                st.stop()


            for lag in range(1, NUM_LAGGED_FEATURES + 1):
                df[f'Lag_{lag}'] = df['Discharge (m³/S)'].shift(lag)
            df.dropna(inplace=True)

        st.success("Data preprocessing complete!")
        pass # <--- INSERTED 'pass' STATEMENT HERE


        # --- Data Exploration ---
        st.subheader("Exploratory Data Analysis")
        data_expander = st.expander("Show Data Exploration", expanded=False)
        with data_expander:
            st.write("Summary Statistics:")
            st.write(df['Discharge (m³/S)'].describe())

            st.line_chart(df, x='Date', y='Discharge (m³/S)', height=300, use_container_width=True)
            st.caption("Time Series plot of Discharge Data")


        # --- Training Parameters ---
        st.header("Step 2: Configure Training Parameters")
        epochs_input, batch_size_input, train_split_input = st.columns(3)

        with epochs_input:
            epochs = st.number_input("🔄 Number of Epochs:", min_value=50, max_value=500, value=DEFAULT_EPOCHS, step=50,
                                    help="Number of training iterations over the entire dataset.")
        with batch_size_input:
            batch_size = st.number_input("📦 Batch Size:", min_value=16, max_value=128, value=DEFAULT_BATCH_SIZE, step=16,
                                        help="Number of samples processed in each training batch.")
        with train_split_input:
            train_split_percent = st.slider("📊 Training Data Percentage:", min_value=60, max_value=90, value=DEFAULT_TRAIN_SPLIT, step=5,
                                            help="Percentage of data to use for training. The rest will be used for testing.")
            train_split = train_split_percent / 100

        # --- Model Training and Testing ---
        model_ops_header = st.header("Step 3: Train and Test Model")
        model_train_button, model_test_button = st.columns(2)


        if model_train_button.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model..."):
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

                model = build_gru_model((X_train.shape[1], X_train.shape[2]))
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0) # Reduced verbosity
                model.save_weights(MODEL_WEIGHTS_PATH)  # Save model weights
            st.success("✅ Model trained and saved!")

            # Display training history (loss curve) - Optional, can add expander
            training_expander = st.expander("Show Training History", expanded=False)
            with training_expander:
                fig_train_hist, ax_train_hist = plt.subplots()
                ax_train_hist.plot(history.history['loss'])
                ax_train_hist.set_title('Model Loss during Training')
                ax_train_hist.set_ylabel('Loss (MSE)')
                ax_train_hist.set_xlabel('Epoch')
                st.pyplot(fig_train_hist)


        if model_test_button.button("🔍 Test Model", use_container_width=True):
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                st.error("❌ No trained model found! Please train the model first.")
                st.stop()

            with st.spinner("Testing model and generating predictions..."):
                # Data Preparation (Scaling and Splitting - same as training for consistency)
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[['Discharge (m³/S)'] + [f'Lag_{i}' for i in range(1, NUM_LAGGED_FEATURES + 1)]])
                train_size = int(len(scaled_data) * train_split)
                train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
                X_train, y_train = train_data[:, 1:], train_data[:, 0]
                X_test, y_test = test_data[:, 1:], test_data[:, 0]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


                model = build_gru_model((X_train.shape[1], X_train.shape[2]))
                model.load_weights(MODEL_WEIGHTS_PATH)  # Load model weights

                # Make predictions
                y_train_pred = model.predict(X_train, verbose=0) # Reduced verbosity
                y_test_pred = model.predict(X_test, verbose=0)   # Reduced verbosity

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

            st.success("✅ Model tested and predictions generated!")

            # --- Display Metrics and Plots ---
            st.header("Step 4: Evaluate Model Performance")
            metrics_expander = st.expander("Show Performance Metrics", expanded=True) # Metrics shown by default
            with metrics_expander:
                st.markdown("### Performance Metrics:")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Training Performance**")
                    st.metric("RMSE (Train)", f"{train_rmse:.4f}", help="Root Mean Squared Error - Lower is better")
                    st.metric("R² (Train)", f"{train_r2:.4f}", help="R-squared - Closer to 1 is better")
                    st.metric("NSE (Train)", f"{train_nse:.4f}", help="Nash-Sutcliffe Efficiency - Closer to 1 is better")

                with col2:
                    st.write("**Testing Performance**")
                    st.metric("RMSE (Test)", f"{test_rmse:.4f}", help="Root Mean Squared Error on unseen data")
                    st.metric("R² (Test)", f"{test_r2:.4f}", help="R-squared on unseen data")
                    st.metric("NSE (Test)", f"{test_nse:.4f}", help="Nash-Sutcliffe Efficiency on unseen data")

                st.markdown("""
                    **Metric Definitions:**
                    * **RMSE (Root Mean Squared Error):** Measures the average magnitude of the errors. Lower values indicate better fit.
                    * **R² (R-squared):** Represents the proportion of variance in the actual data that is predictable from the model. Closer to 1 indicates a better fit.
                    * **NSE (Nash-Sutcliffe Efficiency):**  A normalized statistic that determines the relative magnitude of the residual variance compared to the measured data variance. NSE = 1 corresponds to a perfect match of predicted to observed data. NSE > 0.5 is generally considered acceptable for hydrological models.
                    """)


            plots_expander = st.expander("Show Prediction Plots", expanded=True) # Plots shown by default
            with plots_expander:
                fig, ax = plt.subplots(2, 1, figsize=(10, 8)) # Increased figure height

                # Training plot
                ax[0].plot(df['Date'][-len(y_train_actual)-len(y_test_actual):-len(y_test_actual)], y_train_actual, label="Actual", color="blue") # Correct Date Slicing
                ax[0].plot(df['Date'][-len(y_train_actual)-len(y_test_actual):-len(y_test_actual)], y_train_pred, label="Predicted", color="orange") # Correct Date Slicing
                ax[0].set_title("📈 Training Data: Actual vs. Predicted")
                ax[0].set_ylabel("Discharge (m³/S)")
                ax[0].legend()


                # Testing plot
                ax[1].plot(df['Date'][-len(y_test_actual):], y_test_actual, label="Actual", color="blue") # Correct Date Slicing
                ax[1].plot(df['Date'][-len(y_test_actual):], y_test_pred, label="Predicted", color="orange") # Correct Date Slicing
                ax[1].set_title("📈 Testing Data: Actual vs. Predicted")
                ax[1].set_xlabel("Date") # Added x-axis label
                ax[1].set_ylabel("Discharge (m³/S)")
                ax[1].legend()

                fig.tight_layout(pad=2.0) # Improve spacing
                st.pyplot(fig)


            # --- Download Predictions ---
            download_expander = st.expander("Download Predictions", expanded=False)
            with download_expander:
                results_df = pd.DataFrame({
                    "Date_Train": df['Date'][-len(y_train_actual)-len(y_test_actual):-len(y_train_actual)].tolist() if len(y_train_actual)>0 else np.nan, # Date for training predictions
                    "Actual_Train": y_train_actual.tolist() if len(y_train_actual)>0 else np.nan,
                    "Predicted_Train": y_train_pred.tolist() if len(y_train_pred)>0 else np.nan,
                    "Date_Test": df['Date'][-len(y_test_actual):].tolist() if len(y_test_actual)>0 else np.nan, # Date for test predictions
                    "Actual_Test": y_test_actual.tolist() if len(y_test_actual)>0 else np.nan,
                    "Predicted_Test": y_test_pred.tolist() if len(y_test_pred)>0 else np.nan
                })

                csv_file = results_df.to_csv(index=False)
                st.download_button("📥 Download Predictions as CSV", csv_file, "predictions.csv", "text/csv")

else:  # <----  IMPORTANT:  'else:' MUST BE ALIGNED WITH 'if uploaded_file:' ABOVE. NO SPACES BEFORE 'else:'
    st.info("Please upload a dataset file to begin.")
