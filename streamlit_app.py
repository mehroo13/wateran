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
import seaborn as sns

# -------------------- Model Parameters --------------------
DEFAULT_GRU_UNITS = 64
DEFAULT_DENSE_UNITS = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SPLIT = 80
NUM_LAGGED_FEATURES = 3
MODEL_WEIGHTS_PATH = os.path.join(tempfile.gettempdir(), "gru_model_weights.weights.h5")
MODEL_PLOT_PATH = os.path.join(tempfile.gettempdir(), "gru_model_plot.png")

# -------------------- NSE Function --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

# -------------------- Custom Callback for Epoch Tracking --------------------
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_placeholder, status_placeholder):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_placeholder = progress_placeholder
        self.status_placeholder = status_placeholder

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_placeholder.progress(min(progress, 1.0))
        self.status_placeholder.write(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs.get('loss'):.4f}")

# -------------------- GRU Model --------------------
def build_gru_model(input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate):
    model = tf.keras.Sequential()
    for i in range(gru_layers):
        model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=(i < gru_layers - 1), 
                                      input_shape=input_shape if i == 0 else None))
        model.add(tf.keras.layers.Dropout(0.2))
    for units in dense_units[:dense_layers]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Time Series Prediction", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px;}
    .stButton>button:hover {background-color: #45a049;}
    .stSlider {background-color: #f0f0f0; padding: 10px; border-radius: 8px;}
    .stExpander {background-color: white; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    h1 {color: #2c3e50; font-family: 'Arial';}
    h2, h3 {color: #34495e;}
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Time Series Prediction with GRU")
st.markdown("Design and predict time series data with ease. Enjoy a sleek interface and real-time insights!")

# Initialize session state
for key in ['metrics', 'train_results_df', 'test_results_df', 'fig', 'model_plot']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------- Sidebar for Quick Settings --------------------
with st.sidebar:
    st.header("Quick Settings")
    st.markdown("Adjust key parameters here or dive into details below.")
    epochs = st.slider("Epochs", 10, 1500, DEFAULT_EPOCHS, step=10)
    batch_size = st.slider("Batch Size", 8, 128, DEFAULT_BATCH_SIZE, step=8)
    train_split = st.slider("Training Split (%)", 50, 90, DEFAULT_TRAIN_SPLIT) / 100

# -------------------- Main Layout --------------------
tab1, tab2 = st.tabs(["📊 Data & Model", "📈 Results"])

with tab1:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        with st.expander("📥 Upload Your Data", expanded=True):
            uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"], help="Supports .xlsx files with numeric data.")
            if uploaded_file:
                df = pd.read_excel(uploaded_file)
                st.write("**Preview:**", df.head())
                
                # Datetime Handling
                datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
                date_col = st.selectbox("Select Date Column (Optional):", ["None"] + datetime_cols, index=0)
                if date_col != "None":
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(date_col)
                
                # Variable Selection
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != date_col]
                if len(numeric_cols) < 2:
                    st.error("Need at least two numeric columns!")
                    st.stop()
                output_var = st.selectbox("🎯 Target Variable:", numeric_cols)
                
                # Define input options and default value
                input_options = [col for col in numeric_cols if col != output_var]
                default_input = [numeric_cols[0]] if numeric_cols[0] != output_var else ([input_options[0]] if input_options else [])
                
                input_vars = st.multiselect("🔧 Input Variables:", input_options, default=default_input)
                if not input_vars:
                    st.error("Select at least one input variable!")
                    st.stop()
                
                # Lag Features
                feature_cols = []
                for var in input_vars + [output_var]:
                    for lag in range(1, NUM_LAGGED_FEATURES + 1):
                        df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                        feature_cols.append(f'{var}_Lag_{lag}')
                df.dropna(inplace=True)

    with col2:
        with st.expander("⚙️ Model Architecture", expanded=True):
            gru_layers = st.number_input("GRU Layers", 1, 5, 1)
            gru_units = [st.number_input(f"GRU Layer {i+1} Units", 8, 512, DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") 
                         for i in range(gru_layers)]
            dense_layers = st.number_input("Dense Layers", 1, 5, 1)
            dense_units = [st.number_input(f"Dense Layer {i+1} Units", 8, 512, DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") 
                           for i in range(dense_layers)]
            learning_rate = st.number_input("Learning Rate", 0.00001, 0.1, DEFAULT_LEARNING_RATE, format="%.5f")
            
            if uploaded_file:
                dummy_shape = (1, len(input_vars) + len(feature_cols))
                model = build_gru_model(dummy_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                try:
                    plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True, dpi=96)
                    st.image(MODEL_PLOT_PATH, caption="Model Blueprint", use_container_width=True)
                except ImportError:
                    st.warning("Install 'pydot' and 'graphviz' for model visualization.")

    if uploaded_file:
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
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Train Model"):
                model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                with st.spinner("Training..."):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                                        callbacks=[StreamlitProgressCallback(epochs, progress_bar, status_text)])
                    model.save_weights(MODEL_WEIGHTS_PATH)
                st.success("Training Complete!")
        
        with col_btn2:
            if st.button("🔍 Test Model"):
                if not os.path.exists(MODEL_WEIGHTS_PATH):
                    st.error("Train the model first!")
                else:
                    model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                    model.load_weights(MODEL_WEIGHTS_PATH)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
                    y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
                    y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
                    y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]
                    
                    y_train_pred = np.clip(y_train_pred, 0, None)
                    y_test_pred = np.clip(y_test_pred, 0, None)
                    
                    st.session_state.metrics = {
                        "Training RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                        "Testing RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                        "Training R²": r2_score(y_train_actual, y_train_pred),
                        "Testing R²": r2_score(y_test_actual, y_test_pred),
                        "Training NSE": nse(y_train_actual, y_train_pred),
                        "Testing NSE": nse(y_test_actual, y_test_pred)
                    }
                    st.session_state.train_results_df = pd.DataFrame({f"Actual_{output_var}": y_train_actual, f"Predicted_{output_var}": y_train_pred})
                    st.session_state.test_results_df = pd.DataFrame({f"Actual_{output_var}": y_test_actual, f"Predicted_{output_var}": y_test_pred})
                    
                    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                    sns.lineplot(data=st.session_state.train_results_df, ax=ax[0], palette="tab10", linewidth=2)
                    ax[0].set_title(f"Training: {output_var}", fontsize=14)
                    ax[0].legend()
                    ax[0].grid(True, linestyle='--', alpha=0.5)
                    
                    sns.lineplot(data=st.session_state.test_results_df, ax=ax[1], palette="tab10", linewidth=2)
                    ax[1].set_title(f"Testing: {output_var}", fontsize=14)
                    ax[1].legend()
                    ax[1].grid(True, linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    st.session_state.fig = fig
                    st.success("Testing Complete!")

with tab2:
    if any(st.session_state[key] for key in ['metrics', 'fig', 'train_results_df', 'test_results_df']):
        st.subheader("📊 Results Dashboard")
        if st.session_state.metrics:
            st.write("**Performance Metrics**")
            metrics_df = pd.DataFrame(st.session_state.metrics.items(), columns=["Metric", "Value"]).pivot(columns="Metric", values="Value")
            st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap="Blues"))
        
        col_plot, col_dl = st.columns([3, 1])
        with col_plot:
            if st.session_state.fig:
                st.pyplot(st.session_state.fig)
        
        with col_dl:
            if st.session_state.fig:
                buf = BytesIO()
                st.session_state.fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("⬇️ Download Plot", buf, "prediction_plot.png", "image/png")
            if st.session_state.train_results_df is not None:
                st.download_button("⬇️ Training CSV", st.session_state.train_results_df.to_csv(index=False), "train_predictions.csv")
            if st.session_state.test_results_df is not None:
                st.download_button("⬇️ Testing CSV", st.session_state.test_results_df.to_csv(index=False), "test_predictions.csv")

st.markdown("---")
st.markdown("**Built with ❤️ by xAI | Powered by GRU and Streamlit**")
