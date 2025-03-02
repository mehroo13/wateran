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
        self.progress_placeholder.text(f"Epoch {self.current_epoch}/{self.total_epochs}")

# -------------------- GRU Model with Simplified Structure --------------------
def build_gru_model(input_features, hidden_layers, gru_units_per_layer, output_units=1, learning_rate=0.001):
    model = tf.keras.Sequential()
    
    # Input GRU layer (single timestep with input features)
    model.add(tf.keras.layers.GRU(gru_units_per_layer[0], input_shape=(1, input_features), return_sequences=(hidden_layers > 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Additional hidden GRU layers
    for i in range(1, hidden_layers):
        model.add(tf.keras.layers.GRU(gru_units_per_layer[i], return_sequences=(i < hidden_layers - 1)))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(output_units, activation='linear'))
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# -------------------- Streamlit UI with Cozy Design --------------------
st.set_page_config(page_title="Time Series Research Hub", page_icon="🌱", layout="wide")

# Custom CSS for cozy aesthetics
st.markdown("""
    <style>
    .main {background-color: #f9f1e9;}
    .stButton>button {background-color: #d4a373; color: white; border-radius: 8px;}
    .stSlider>div>div>div {background-color: #e8b923;}
    .stExpander {background-color: #fff7ed; border-radius: 10px;}
    h1 {color: #8b5e34; font-family: 'Georgia', serif;}
    h2, h3 {color: #a6764e; font-family: 'Georgia', serif;}
    .stTable {background-color: #fff7ed; padding: 10px; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Time Series Research Hub with GRU")
st.markdown("Discover insights from your time series data with a warm, intuitive GRU-based tool designed for researchers.")

# Initialize session state
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("📂 Upload Your Data", expanded=True):
        uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"], help="Ensure your data has numeric columns!")

with col2:
    with st.expander("🛠️ Model Configuration", expanded=True):
        epochs = st.slider("Epochs", 10, 500, DEFAULT_EPOCHS, step=10, help="Number of training iterations")
        batch_size = st.slider("Batch Size", 8, 128, DEFAULT_BATCH_SIZE, step=8, help="Samples per gradient update")
        train_split = st.slider("Train Split (%)", 50, 90, DEFAULT_TRAIN_SPLIT, help="Percentage of data for training") / 100

        st.subheader("GRU Architecture")
        hidden_layers = st.number_input("Hidden GRU Layers", min_value=1, max_value=3, value=1, step=1, help="Number of GRU layers")
        gru_units_per_layer = []
        for i in range(hidden_layers):
            units = st.number_input(f"Layer {i+1} Units", min_value=16, max_value=256, value=DEFAULT_GRU_UNITS, step=16, key=f"gru_{i}")
            gru_units_per_layer.append(units)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, DEFAULT_LEARNING_RATE, step=0.0001, format="%.5f", help="Step size for optimization")

# Process data if uploaded
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    with col1:
        st.write("**Data Preview:**", df.head(5))

        # Datetime handling
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        date_col = st.selectbox("Datetime Column (optional)", ["None"] + datetime_cols, index=0)
        if date_col != "None":
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)

        # Numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != date_col]
        if len(numeric_cols) < 2:
            st.error("Please upload a dataset with at least two numeric columns.")
            st.stop()

        output_var = st.selectbox("🎯 Target Variable", numeric_cols)
        input_vars = st.multiselect("🔍 Input Variables", [col for col in numeric_cols if col != output_var], default=[numeric_cols[0]])
        if not input_vars:
            st.error("Select at least one input variable.")
            st.stop()

        # Create lagged features
        feature_cols = []
        for var in input_vars + [output_var]:
            for lag in range(1, NUM_LAGGED_FEATURES + 1):
                df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                feature_cols.append(f'{var}_Lag_{lag}')
        df.dropna(inplace=True)

    with col2:
        st.write(f"**Training Rows:** {int(len(df) * train_split)} | **Testing Rows:** {len(df) - int(len(df) * train_split)}")
        
        # Model visualization
        input_features = len(input_vars) + len(feature_cols)
        model = build_gru_model(input_features, hidden_layers, gru_units_per_layer, learning_rate=learning_rate)
        try:
            plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True)
            st.image(MODEL_PLOT_PATH, caption="Your GRU Model", use_container_width=True)
        except:
            st.warning("Install 'pydot' and 'graphviz' for model visualization.")

    # Data preparation
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
            model = build_gru_model(input_features, hidden_layers, gru_units_per_layer, learning_rate=learning_rate)
            with st.spinner("Training your model..."):
                progress_placeholder = st.empty()
                callback = StreamlitProgressCallback(epochs, progress_placeholder)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[callback])
                model.save_weights(MODEL_WEIGHTS_PATH)
            st.success("Training completed!")

    with col_btn2:
        if st.button("🔍 Evaluate Model"):
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                st.error("Please train the model first!")
                st.stop()
            model = build_gru_model(input_features, hidden_layers, gru_units_per_layer, learning_rate=learning_rate)
            model.load_weights(MODEL_WEIGHTS_PATH)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Inverse transform predictions
            y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
            y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
            y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
            y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

            # Metrics
            metrics = {
                "Train RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                "Test RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                "Train R²": r2_score(y_train_actual, y_train_pred),
                "Test R²": r2_score(y_test_actual, y_test_pred),
                "Train NSE": nse(y_train_actual, y_train_pred),
                "Test NSE": nse(y_test_actual, y_test_pred)
            }
            st.session_state.metrics = metrics
            st.session_state.results_df = pd.DataFrame({
                "Train_Actual": y_train_actual, "Train_Predicted": y_train_pred,
                "Test_Actual": y_test_actual, "Test_Predicted": y_test_pred
            })

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_train_actual, label="Train Actual", color="#8b5e34")
            ax.plot(y_train_pred, label="Train Predicted", color="#d4a373", linestyle="--")
            ax.plot(range(len(y_train_actual), len(df)), y_test_actual, label="Test Actual", color="#8b5e34")
            ax.plot(range(len(y_train_actual), len(df)), y_test_pred, label="Test Predicted", color="#d4a373", linestyle="--")
            ax.set_title(f"{output_var} Predictions", fontsize=16)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            st.session_state.fig = fig
            st.success("Evaluation completed!")

# Results Section
if st.session_state.metrics or st.session_state.fig:
    with st.expander("📊 Research Insights", expanded=True):
        if st.session_state.metrics:
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(st.session_state.metrics.items(), columns=["Metric", "Value"])
            st.table(metrics_df.style.format({"Value": "{:.4f}"}))

        if st.session_state.fig:
            st.subheader("Prediction Visualization")
            st.pyplot(st.session_state.fig)
            buf = BytesIO()
            st.session_state.fig.savefig(buf, format="png", dpi=300)
            st.download_button("⬇️ Download Plot", buf.getvalue(), "predictions.png", "image/png")

            if st.session_state.results_df is not None:
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button("⬇️ Download Results", csv, "results.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("**Crafted with care by xAI | Powered by GRU & Streamlit 🌟**")
