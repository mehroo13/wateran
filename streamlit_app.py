import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO
import time
from tensorflow.keras.utils import plot_model

# -------------------- Model Parameters --------------------
DEFAULT_GRU_UNITS = 64
DEFAULT_DENSE_UNITS = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SPLIT = 80  # Percentage of data used for training
NUM_LAGGED_FEATURES = 3  # Number of lag features
MODEL_WEIGHTS_PATH = os.path.join(tempfile.gettempdir(), "gru_model_weights.weights.h5")
MODEL_PLOT_PATH = os.path.join(tempfile.gettempdir(), "gru_model_plot.png")

# -------------------- NSE Function --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

# -------------------- Custom Callback for Epoch Tracking --------------------
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_placeholder):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_placeholder = progress_placeholder
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        progress = self.current_epoch / self.total_epochs
        self.progress_placeholder.progress(min(progress, 1.0))
        self.progress_placeholder.text(f"Epoch {self.current_epoch}/{self.total_epochs} completed")

# -------------------- GRU Model with Custom Layers --------------------
def build_gru_model(input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate):
    model = tf.keras.Sequential()
    
    # Add GRU layers
    for i in range(gru_layers):
        if i == 0:
            model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=(i < gru_layers - 1), input_shape=input_shape))
        else:
            model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=(i < gru_layers - 1)))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Add Dense layers
    for units in dense_units[:dense_layers]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Time Series Prediction", page_icon="📈", layout="wide")
st.title("🌊 Time Series Prediction with GRU")
st.markdown("**Design and predict time series data with a customizable GRU model. Visualize your architecture and results in real-time!**")

# Initialize session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'train_results_df' not in st.session_state:
    st.session_state.train_results_df = None
if 'test_results_df' not in st.session_state:
    st.session_state.test_results_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'model_plot' not in st.session_state:
    st.session_state.model_plot = None

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("📥 Upload Your Data", expanded=True):
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

with col2:
    with st.expander("⚙️ Model Settings", expanded=True):
        epochs = st.slider("Epochs:", 1, 1500, DEFAULT_EPOCHS, step=10)
        batch_size = st.slider("Batch Size:", 8, 128, DEFAULT_BATCH_SIZE, step=8)
        train_split = st.slider("Training Data %:", 50, 90, DEFAULT_TRAIN_SPLIT) / 100

        # Custom model architecture
        st.subheader("Model Architecture")
        gru_layers = st.number_input("Number of GRU Layers:", min_value=1, max_value=5, value=1, step=1)
        gru_units = []
        for i in range(gru_layers):
            units = st.number_input(f"GRU Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}")
            gru_units.append(units)

        dense_layers = st.number_input("Number of Dense Layers:", min_value=1, max_value=5, value=1, step=1)
        dense_units = []
        for i in range(dense_layers):
            units = st.number_input(f"Dense Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}")
            dense_units.append(units)

        learning_rate = st.number_input("Learning Rate:", min_value=0.00001, max_value=0.1, value=DEFAULT_LEARNING_RATE, format="%.5f")

# Process data if uploaded
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    with col1:
        st.write("**Dataset Preview:**", df.head())

        # Datetime column handling
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        if datetime_cols:
            date_col = st.selectbox("Select datetime column (optional):", ["None"] + datetime_cols, index=0)
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            else:
                st.info("Using index for ordering (assuming sequential data).")
        else:
            st.info("No datetime column found. Using index for ordering.")

        # Numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (datetime_cols and col != date_col or True)]
        if len(numeric_cols) < 2:
            st.error("Dataset needs at least two numeric columns.")
            st.stop()

        # Variable selection
        output_var = st.selectbox("🎯 Output Variable to Predict:", numeric_cols)
        input_vars = st.multiselect("🔧 Input Variables:", [col for col in numeric_cols if col != output_var], default=[col for col in numeric_cols if col != output_var][:1])
        if not input_vars:
            st.error("Select at least one input variable.")
            st.stop()

        # Lag features for all variables
        feature_cols = []
        for var in input_vars + [output_var]:
            for lag in range(1, NUM_LAGGED_FEATURES + 1):
                df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                feature_cols.append(f'{var}_Lag_{lag}')
        df.dropna(inplace=True)

    with col2:
        st.write(f"**Training Size:** {int(len(df) * train_split)} rows | **Testing Size:** {len(df) - int(len(df) * train_split)} rows")

        # Visualize model structure with error handling
        dummy_input_shape = (1, len(input_vars) + len(feature_cols))  # Timesteps=1, features from input_vars + lags
        model = build_gru_model(
            input_shape=dummy_input_shape,
            gru_layers=gru_layers,
            dense_layers=dense_layers,
            gru_units=gru_units,
            dense_units=dense_units,
            learning_rate=learning_rate
        )
        try:
            plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True, dpi=96)
            st.image(MODEL_PLOT_PATH, caption="GRU Model Structure", use_column_width=True)
        except ImportError:
            st.warning("Model visualization requires 'pydot' and 'graphviz'. Install them with 'pip install pydot graphviz' and ensure Graphviz is installed on your system.")

    # Data processing
    train_size = int(len(df) * train_split)
    train_df, test_df = df[:train_size], df[train_size:]
    all_feature_cols = input_vars + feature_cols

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[output_var] + all_feature_cols])
    test_scaled = scaler.transform(test_df[[output_var] + all_feature_cols])

    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Train Model"):
            model = build_gru_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                gru_layers=gru_layers,
                dense_layers=dense_layers,
                gru_units=gru_units,
                dense_units=dense_units,
                learning_rate=learning_rate
            )
            try:
                with st.spinner("Training in progress..."):
                    progress_placeholder = st.empty()
                    callback = StreamlitProgressCallback(epochs, progress_placeholder)
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[callback]
                    )
                    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
                    model.save_weights(MODEL_WEIGHTS_PATH)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    with col_btn2:
        if st.button("🔍 Test Model"):
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                st.error("Train the model first!")
                st.stop()
            model = build_gru_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                gru_layers=gru_layers,
                dense_layers=dense_layers,
                gru_units=gru_units,
                dense_units=dense_units,
                learning_rate=learning_rate
            )
            try:
                model.load_weights(MODEL_WEIGHTS_PATH)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Inverse transform and clamp negative values to zero
                y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
                y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
                y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
                y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

                # Ensure no negative predictions
                y_train_pred = np.clip(y_train_pred, 0, None)
                y_test_pred = np.clip(y_test_pred, 0, None)

                # Metrics
                metrics = {
                    "Training RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                    "Testing RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                    "Training R²": r2_score(y_train_actual, y_train_pred),
                    "Testing R²": r2_score(y_test_actual, y_test_pred),
                    "Training NSE": nse(y_train_actual, y_train_pred),
                    "Testing NSE": nse(y_test_actual, y_test_pred)
                }
                # Internal check: Test > Train performance (not displayed)
                if metrics["Testing RMSE"] < metrics["Training RMSE"] or metrics["Testing R²"] > metrics["Training R²"]:
                    pass  # Silent check

                # Store results
                st.session_state.metrics = metrics
                st.session_state.train_results_df = pd.DataFrame({
                    f"Actual_{output_var}": y_train_actual,
                    f"Predicted_{output_var}": y_train_pred
                })
                st.session_state.test_results_df = pd.DataFrame({
                    f"Actual_{output_var}": y_test_actual,
                    f"Predicted_{output_var}": y_test_pred
                })

                # Plot
                fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                ax[0].plot(y_train_actual, label="Actual", color="#1f77b4", linewidth=2)
                ax[0].plot(y_train_pred, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
                ax[0].set_title(f"Training Data: {output_var}", fontsize=14, pad=10)
                ax[0].legend()
                ax[0].grid(True, linestyle='--', alpha=0.7)

                ax[1].plot(y_test_actual, label="Actual", color="#1f77b4", linewidth=2)
                ax[1].plot(y_test_pred, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
                ax[1].set_title(f"Testing Data: {output_var}", fontsize=14, pad=10)
                ax[1].legend()
                ax[1].grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.session_state.fig = fig
                st.success("Model tested successfully!")
            except Exception as e:
                st.error(f"Testing failed: {str(e)}")

# Results Section
if st.session_state.metrics or st.session_state.fig or st.session_state.train_results_df or st.session_state.test_results_df:
    with st.expander("📊 Results", expanded=True):
        # Metrics Table
        if st.session_state.metrics:
            st.subheader("📏 Model Performance Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["RMSE", "R²", "NSE"],
                "Training": [f"{st.session_state.metrics['Training RMSE']:.4f}", 
                             f"{st.session_state.metrics['Training R²']:.4f}", 
                             f"{st.session_state.metrics['Training NSE']:.4f}"],
                "Testing": [f"{st.session_state.metrics['Testing RMSE']:.4f}", 
                            f"{st.session_state.metrics['Testing R²']:.4f}", 
                            f"{st.session_state.metrics['Testing NSE']:.4f}"]
            })
            st.table(metrics_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]}
            ]))

        # Plot and Download
        col_plot, col_dl = st.columns([3, 1])
        with col_plot:
            if st.session_state.fig:
                st.subheader("📈 Prediction Plots")
                st.pyplot(st.session_state.fig)

        with col_dl:
            if st.session_state.fig:
                buf = BytesIO()
                st.session_state.fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button("⬇️ Download Plot", buf, "prediction_plot.png", "image/png", key="plot_dl")

            if st.session_state.train_results_df is not None:
                train_csv = st.session_state.train_results_df.to_csv(index=False)
                st.download_button("⬇️ Training Data CSV", train_csv, "train_predictions.csv", "text/csv", key="train_dl")

            if st.session_state.test_results_df is not None:
                test_csv = st.session_state.test_results_df.to_csv(index=False)
                st.download_button("⬇️ Testing Data CSV", test_csv, "test_predictions.csv", "text/csv", key="test_dl")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ by xAI | Powered by GRU and Streamlit**")
