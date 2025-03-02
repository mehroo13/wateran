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
        self.status_placeholder.markdown(f"**Epoch {epoch + 1}/{self.total_epochs}** - Loss: `{logs.get('loss'):.4f}`")

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
    .main {background: linear-gradient(135deg, #e0f7fa, #b2ebf2); padding: 20px;}
    .stButton>button {background: #0288d1; color: white; border-radius: 12px; padding: 12px 24px; font-weight: bold; transition: all 0.3s;}
    .stButton>button:hover {background: #0277bd; transform: scale(1.05);}
    .stSlider {background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);}
    .stExpander {background: #ffffff; border-radius: 15px; box-shadow: 0 6px 15px rgba(0,0,0,0.08); padding: 10px;}
    h1 {color: #01579b; font-family: 'Segoe UI', sans-serif; text-align: center; font-size: 2.5em; margin-bottom: 0;}
    h2 {color: #0288d1; font-family: 'Segoe UI', sans-serif; font-size: 1.8em;}
    h3 {color: #039be5; font-family: 'Segoe UI', sans-serif;}
    .stProgress .st-bo {background: #4fc3f7;}
    .metric-box {background: #e1f5fe; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;}
    .download-btn {margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Time Series Prediction with GRU")
st.markdown("<p style='text-align: center; color: #0277bd;'>Unleash the power of GRU models with a stunning, interactive experience!</p>", unsafe_allow_html=True)

# Initialize session state
for key in ['metrics', 'train_results_df', 'test_results_df', 'fig', 'model_plot']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------- Main Layout --------------------
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    with st.expander("📥 Upload Your Data", expanded=True):
        st.markdown("**Step 1: Load Your Data**")
        uploaded_file = st.file_uploader("Drop an Excel file here", type=["xlsx"], help="Supports .xlsx with numeric data", label_visibility="collapsed")
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.markdown("**Data Sneak Peek:**")
            st.dataframe(df.head(), use_container_width=True)
            
            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
            date_col = st.selectbox("📅 Select Date Column (Optional):", ["None"] + datetime_cols, index=0, help="Sorts data by this column if selected.")
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            else:
                st.info("No date column selected. Assuming sequential data.")
            
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != date_col]
            if len(numeric_cols) < 2:
                st.error("🚨 Your dataset needs at least two numeric columns!")
                st.stop()
            
            output_var = st.selectbox("🎯 Target Variable:", numeric_cols, help="What you want to predict.")
            input_options = [col for col in numeric_cols if col != output_var]
            default_input = [numeric_cols[0]] if numeric_cols[0] != output_var else ([input_options[0]] if input_options else [])
            input_vars = st.multiselect("🔧 Input Variables:", input_options, default=default_input, help="Features to train the model.")
            if not input_vars:
                st.error("🚨 Select at least one input variable!")
                st.stop()
            
            feature_cols = []
            for var in input_vars + [output_var]:
                for lag in range(1, NUM_LAGGED_FEATURES + 1):
                    df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                    feature_cols.append(f'{var}_Lag_{lag}')
            df.dropna(inplace=True)

with col2:
    with st.expander("⚙️ Model Settings", expanded=True):
        st.markdown("**Step 2: Tune Your Model**")
        epochs = st.slider("⏳ Epochs:", 1, 1500, DEFAULT_EPOCHS, step=10, help="Number of training iterations.")
        batch_size = st.slider("📦 Batch Size:", 8, 128, DEFAULT_BATCH_SIZE, step=8, help="Samples per gradient update.")
        train_split = st.slider("📊 Training Data %:", 50, 90, DEFAULT_TRAIN_SPLIT, help="Percentage of data for training.") / 100

        st.markdown("### 🛠️ Model Architecture")
        gru_layers = st.number_input("GRU Layers:", min_value=1, max_value=5, value=1, step=1, help="Number of GRU layers.")
        gru_units = [st.number_input(f"GRU Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") 
                     for i in range(gru_layers)]

        dense_layers = st.number_input("Dense Layers:", min_value=1, max_value=5, value=1, step=1, help="Number of Dense layers.")
        dense_units = [st.number_input(f"Dense Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") 
                       for i in range(dense_layers)]

        learning_rate = st.number_input("🚀 Learning Rate:", min_value=0.00001, max_value=0.1, value=DEFAULT_LEARNING_RATE, format="%.5f", help="Step size for optimization.")

        if uploaded_file:
            dummy_input_shape = (1, len(input_vars) + len(feature_cols))
            model = build_gru_model(dummy_input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate)
            try:
                plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True, dpi=96)
                st.image(MODEL_PLOT_PATH, caption="Your Model Blueprint", use_container_width=True)
            except ImportError:
                st.warning("🛠️ Install 'pydot' and 'graphviz' to see the model structure!")

# Process data and buttons
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

    with col2:
        st.markdown(f"**Training Size:** `{train_size}` rows | **Testing Size:** `{len(df) - train_size}` rows", unsafe_allow_html=True)

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Train Model", use_container_width=True):
            model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
            with st.spinner("Training your model..."):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
                                    callbacks=[StreamlitProgressCallback(epochs, progress_bar, status_text)])
                os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
                model.save_weights(MODEL_WEIGHTS_PATH)
            st.success("🎉 Model trained successfully!")

    with col_btn2:
        if st.button("🔍 Test Model", use_container_width=True):
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                st.error("🚨 Please train the model first!")
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

                fig, ax = plt.subplots(2, 1, figsize=(12, 8), facecolor='#f0f4f8')
                sns.lineplot(data=st.session_state.train_results_df, ax=ax[0], palette="deep", linewidth=2.5)
                ax[0].set_title(f"Training Data: {output_var}", fontsize=16, color='#01579b')
                ax[0].legend(fontsize=12)
                ax[0].grid(True, linestyle='--', alpha=0.3)
                ax[0].set_facecolor('#ffffff')

                sns.lineplot(data=st.session_state.test_results_df, ax=ax[1], palette="deep", linewidth=2.5)
                ax[1].set_title(f"Testing Data: {output_var}", fontsize=16, color='#01579b')
                ax[1].legend(fontsize=12)
                ax[1].grid(True, linestyle='--', alpha=0.3)
                ax[1].set_facecolor('#ffffff')

                plt.tight_layout()
                st.session_state.fig = fig
                st.success("🎉 Model tested successfully!")

# Results Section
if any(st.session_state[key] for key in ['metrics', 'fig', 'train_results_df', 'test_results_df']):
    with st.expander("📊 Results", expanded=True):
        st.subheader("📈 Your Model's Performance")
        if st.session_state.metrics:
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.markdown(f"<div class='metric-box'><strong>Training RMSE</strong><br>{st.session_state.metrics['Training RMSE']:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><strong>Training R²</strong><br>{st.session_state.metrics['Training R²']:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><strong>Training NSE</strong><br>{st.session_state.metrics['Training NSE']:.4f}</div>", unsafe_allow_html=True)
            with col_metrics2:
                st.markdown(f"<div class='metric-box'><strong>Testing RMSE</strong><br>{st.session_state.metrics['Testing RMSE']:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><strong>Testing R²</strong><br>{st.session_state.metrics['Testing R²']:.4f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box'><strong>Testing NSE</strong><br>{st.session_state.metrics['Testing NSE']:.4f}</div>", unsafe_allow_html=True)

        col_plot, col_dl = st.columns([3, 1])
        with col_plot:
            if st.session_state.fig:
                st.pyplot(st.session_state.fig)

        with col_dl:
            if st.session_state.fig:
                buf = BytesIO()
                st.session_state.fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("⬇️ Download Plot", buf, "prediction_plot.png", "image/png", key="plot_dl", use_container_width=True)
            if st.session_state.train_results_df is not None:
                st.download_button("⬇️ Training CSV", st.session_state.train_results_df.to_csv(index=False), "train_predictions.csv", key="train_dl", use_container_width=True)
            if st.session_state.test_results_df is not None:
                st.download_button("⬇️ Testing CSV", st.session_state.test_results_df.to_csv(index=False), "test_predictions.csv", key="test_dl", use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #01579b;'>Built with ❤️ by xAI | Powered by GRU and Streamlit</p>", unsafe_allow_html=True)
