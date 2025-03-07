import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from io import BytesIO
from tensorflow.keras.utils import plot_model
import plotly.graph_objects as go

# -------------------- Model Parameters --------------------
DEFAULT_GRU_UNITS = 64
DEFAULT_LSTM_UNITS = 64
DEFAULT_RNN_UNITS = 64
DEFAULT_DENSE_UNITS = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SPLIT = 80
DEFAULT_NUM_LAGS = 3
MODEL_WEIGHTS_PATH = os.path.join(tempfile.gettempdir(), "model_weights.weights.h5")
MODEL_FULL_PATH = os.path.join(tempfile.gettempdir(), "model.h5")
MODEL_PLOT_PATH = os.path.join(tempfile.gettempdir(), "model_plot.png")

# -------------------- Metric Functions --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

def kge(actual, predicted):
    r = np.corrcoef(actual, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(actual)
    beta = np.mean(predicted) / np.mean(actual)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

all_metrics_dict = {
    "RMSE": lambda a, p: np.sqrt(mean_squared_error(a, p)),
    "MAE": lambda a, p: mean_absolute_error(a, p),
    "R²": lambda a, p: r2_score(a, p),
    "NSE": nse,
    "KGE": kge,
}

# -------------------- Custom Callbacks --------------------
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

# -------------------- Model Definition --------------------
def build_model(input_shape, model_type, layers, units, dense_layers, dense_units, learning_rate):
    model = tf.keras.Sequential()
    
    if model_type == "Hybrid":
        selected_models = st.session_state.hybrid_models
        if not selected_models:
            st.error("No hybrid models selected. Defaulting to GRU.")
            selected_models = ["GRU"]
        
        total_layers = layers
        layers_per_model = max(1, total_layers // len(selected_models))
        
        for idx, sub_model in enumerate(selected_models):
            sub_units = units[:layers_per_model] if len(units) >= layers_per_model else units + [units[-1]] * (layers_per_model - len(units))
            for i in range(layers_per_model):
                return_seq = not (idx == len(selected_models) - 1 and i == layers_per_model - 1)
                if sub_model == "GRU":
                    model.add(tf.keras.layers.GRU(sub_units[i], return_sequences=return_seq, 
                                                  input_shape=input_shape if idx == 0 and i == 0 else None))
                elif sub_model == "LSTM":
                    model.add(tf.keras.layers.LSTM(sub_units[i], return_sequences=return_seq, 
                                                   input_shape=input_shape if idx == 0 and i == 0 else None))
                elif sub_model == "RNN":
                    model.add(tf.keras.layers.SimpleRNN(sub_units[i], return_sequences=return_seq, 
                                                        input_shape=input_shape if idx == 0 and i == 0 else None))
                model.add(tf.keras.layers.Dropout(0.3))
    else:
        for i in range(layers):
            return_seq = i < layers - 1
            if model_type == "GRU":
                model.add(tf.keras.layers.GRU(units[i], return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
            elif model_type == "LSTM":
                model.add(tf.keras.layers.LSTM(units[i], return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
            elif model_type == "RNN":
                model.add(tf.keras.layers.SimpleRNN(units[i], return_sequences=return_seq, input_shape=input_shape if i == 0 else None))
            model.add(tf.keras.layers.Dropout(0.3))
    
    for unit in dense_units[:dense_layers]:
        model.add(tf.keras.layers.Dense(unit, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='relu'))  # Ensure non-negative output
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# -------------------- Styling and Streamlit UI --------------------
st.set_page_config(page_title="Wateran", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #007bff; color: white; border-radius: 8px; padding: 10px 20px; font-weight: bold; }
    .stButton>button:hover { background-color: #0056b3; }
    .metric-box { background-color: #ffffff; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Wateran: Time Series Prediction")
st.markdown("**Simple, Fast, and Accurate Predictions Powered by Neural Networks**", unsafe_allow_html=True)

# Initialize session state
if 'model_type' not in st.session_state:
    st.session_state.model_type = "GRU"
if 'num_lags' not in st.session_state:
    st.session_state.num_lags = DEFAULT_NUM_LAGS
for key in ['metrics', 'train_results_df', 'test_results_df', 'fig', 'scaler', 'input_vars', 'output_var', 
            'gru_layers', 'lstm_layers', 'rnn_layers', 'gru_units', 'lstm_units', 'rnn_units', 'dense_layers', 
            'dense_units', 'learning_rate', 'feature_cols', 'selected_metrics', 'var_types', 'date_col', 
            'df', 'cv_metrics', 'X_train', 'y_train', 'X_test', 'y_test', 'model', 'hybrid_models']:
    if key not in st.session_state:
        if key in ['gru_layers', 'lstm_layers', 'rnn_layers', 'dense_layers']:
            st.session_state[key] = 1
        elif key == 'gru_units':
            st.session_state[key] = [DEFAULT_GRU_UNITS]
        elif key == 'lstm_units':
            st.session_state[key] = [DEFAULT_LSTM_UNITS]
        elif key == 'rnn_units':
            st.session_state[key] = [DEFAULT_RNN_UNITS]
        elif key == 'dense_units':
            st.session_state[key] = [DEFAULT_DENSE_UNITS]
        elif key == 'learning_rate':
            st.session_state[key] = DEFAULT_LEARNING_RATE
        elif key == 'hybrid_models':
            st.session_state[key] = ["GRU"]
        else:
            st.session_state[key] = None

# Sidebar
with st.sidebar:
    st.header("Navigation")
    st.button("📥 Data Input", key="nav_data")
    st.button("⚙️ Model Configuration", key="nav_config")
    st.button("📊 Results", key="nav_results")
    st.button("🔮 New Predictions", key="nav_predict")
    with st.expander("ℹ️ Help"):
        st.markdown("""
        - **Layers**: Recurrent layers (1-5 recommended).
        - **Dense Layers**: Fully connected layers.
        - **Dynamic Variables**: Lagged values.
        - **Static Variables**: No lags.
        - **Metrics**: NSE/KGE ideal = 1, RMSE/MAE ideal = 0.
        """)

# Main Layout
col1, col2 = st.columns([2, 1], gap="large")

# Left Column: Data Input
with col1:
    st.subheader("📥 Data Input", divider="blue")
    uploaded_file = st.file_uploader("Upload Training Data (Excel)", type=["xlsx"], key="train_data")
    
    if uploaded_file:
        @st.cache_data
        def load_data(file):
            return pd.read_excel(file)
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.markdown("**Dataset Preview:**")
        st.dataframe(df.head(5), use_container_width=True)
        
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        date_col = st.selectbox("Select Date Column (optional)", ["None"] + datetime_cols, index=0, key="date_col_train")
        st.session_state.date_col = date_col if date_col != "None" else None
        if date_col != "None":
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != st.session_state.date_col]
        if len(numeric_cols) < 2:
            st.error("Dataset requires at least two numeric columns.")
            st.stop()
        
        st.markdown("**Variable Selection**")
        output_var = st.selectbox("🎯 Output Variable", numeric_cols, key="output_var_train")
        available_input_cols = [col for col in numeric_cols if col != output_var]
        input_vars = st.multiselect("🔧 Input Variables", available_input_cols, default=[available_input_cols[0]], key="input_vars_train")
        if not input_vars:
            st.error("Select at least one input variable.")
            st.stop()

        with st.expander("Variable Types", expanded=True):
            var_types = {}
            for var in input_vars:
                var_types[var] = st.selectbox(f"{var} Type", ["Dynamic", "Static"], key=f"{var}_type")
            st.session_state.var_types = var_types
        
        st.session_state.input_vars = input_vars
        st.session_state.output_var = output_var

# Right Column: Model Configuration
with col2:
    st.subheader("⚙️ Model Configuration", divider="blue")
    
    model_type = st.selectbox("Model Type", ["GRU", "LSTM", "RNN", "Hybrid"], index=0, key="model_type_select")
    st.session_state.model_type = model_type
    
    st.markdown("**Training Parameters**")
    # Use session state directly in the widget to avoid reassignment issues
    st.session_state.num_lags = st.number_input("Number of Lags", min_value=1, max_value=10, value=st.session_state.num_lags, step=1, key="num_lags")
    
    epochs = st.slider("Epochs", 1, 1500, DEFAULT_EPOCHS, step=10, key="epochs")
    batch_size = st.slider("Batch Size", 8, 128, DEFAULT_BATCH_SIZE, step=8, key="batch_size")
    train_split = st.slider("Training Data %", 50, 90, DEFAULT_TRAIN_SPLIT, key="train_split") / 100
    
    with st.expander("Model Architecture", expanded=False):
        if model_type == "Hybrid":
            hybrid_models = st.multiselect("Select Hybrid Models", ["GRU", "LSTM", "RNN"], default=st.session_state.hybrid_models, key="hybrid_models_select")
            st.session_state.hybrid_models = hybrid_models if hybrid_models else ["GRU"]
            hybrid_layers = st.number_input("Total Hybrid Layers", min_value=1, max_value=10, value=st.session_state.gru_layers, step=1, key="hybrid_layers")
            st.session_state.gru_layers = hybrid_layers
            st.session_state.gru_units = [st.number_input(f"Hybrid Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"hybrid_{i}") for i in range(hybrid_layers)]
        elif model_type == "GRU":
            st.session_state.gru_layers = st.number_input("GRU Layers", min_value=1, max_value=5, value=st.session_state.gru_layers, step=1, key="gru_layers")
            st.session_state.gru_units = [st.number_input(f"GRU Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") for i in range(st.session_state.gru_layers)]
        elif model_type == "LSTM":
            st.session_state.lstm_layers = st.number_input("LSTM Layers", min_value=1, max_value=5, value=st.session_state.lstm_layers, step=1, key="lstm_layers")
            st.session_state.lstm_units = [st.number_input(f"LSTM Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_LSTM_UNITS, step=8, key=f"lstm_{i}") for i in range(st.session_state.lstm_layers)]
        elif model_type == "RNN":
            st.session_state.rnn_layers = st.number_input("RNN Layers", min_value=1, max_value=5, value=st.session_state.rnn_layers, step=1, key="rnn_layers")
            st.session_state.rnn_units = [st.number_input(f"RNN Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_RNN_UNITS, step=8, key=f"rnn_{i}") for i in range(st.session_state.rnn_layers)]
        
        st.session_state.dense_layers = st.number_input("Dense Layers", min_value=1, max_value=5, value=st.session_state.dense_layers, step=1, key="dense_layers")
        st.session_state.dense_units = [st.number_input(f"Dense Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") for i in range(st.session_state.dense_layers)]
        st.session_state.learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=st.session_state.learning_rate, format="%.5f", key="learning_rate")
    
    st.markdown("**Evaluation Metrics**")
    all_metrics = ["RMSE", "MAE", "R²", "NSE", "KGE"]
    st.session_state.selected_metrics = st.multiselect("Select Metrics", all_metrics, default=all_metrics, key="metrics_select") or all_metrics

    if uploaded_file:
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🚀 Train Model", key="train_button"):
                df = st.session_state.df.copy()
                feature_cols = []
                num_lags = st.session_state.num_lags
                for var in st.session_state.input_vars:
                    if st.session_state.var_types[var] == "Dynamic":
                        for lag in range(1, num_lags + 1):
                            df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                            feature_cols.append(f'{var}_Lag_{lag}')
                    else:
                        feature_cols.append(var)
                for lag in range(1, num_lags + 1):
                    df[f'{st.session_state.output_var}_Lag_{lag}'] = df[st.session_state.output_var].shift(lag)
                    feature_cols.append(f'{st.session_state.output_var}_Lag_{lag}')
                
                df = df.dropna()
                st.session_state.feature_cols = feature_cols
                
                train_size = int(len(df) * train_split)
                train_df, test_df = df[:train_size], df[train_size:]
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(train_df[feature_cols + [st.session_state.output_var]])
                test_scaled = scaler.transform(test_df[feature_cols + [st.session_state.output_var]])
                st.session_state.scaler = scaler
                
                X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
                X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                st.session_state.X_train, st.session_state.y_train = X_train, y_train
                st.session_state.X_test, st.session_state.y_test = X_test, y_test

                layers = (st.session_state.gru_layers if model_type in ["GRU", "Hybrid"] else 
                          st.session_state.lstm_layers if model_type == "LSTM" else 
                          st.session_state.rnn_layers)
                units = (st.session_state.gru_units if model_type in ["GRU", "Hybrid"] else 
                         st.session_state.lstm_units if model_type == "LSTM" else 
                         st.session_state.rnn_units)
                st.session_state.model = build_model(
                    (X_train.shape[1], X_train.shape[2]), model_type, layers, units, 
                    st.session_state.dense_layers, st.session_state.dense_units, st.session_state.learning_rate
                )
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                with st.spinner("Training in progress..."):
                    progress_placeholder = st.empty()
                    callback = StreamlitProgressCallback(epochs, progress_placeholder)
                    st.session_state.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                              validation_data=(X_test, y_test), callbacks=[callback, early_stopping, lr_scheduler], verbose=1)
                    st.session_state.model.save_weights(MODEL_WEIGHTS_PATH)
                    st.session_state.model.save(MODEL_FULL_PATH)
                    st.success("Model trained and saved successfully!")
        
        with col_btn2:
            if st.button(f"🤖 Suggest {model_type} Units", key="suggest_button"):
                if "X_train" not in st.session_state:
                    st.error("Please train the model first.")
                else:
                    st.info(f"Suggested {model_type} Units: [64, 32]")

        with col_btn3:
            if st.button("🔍 Test Model", key="test_button"):
                if not os.path.exists(MODEL_WEIGHTS_PATH):
                    st.error("Train the model first!")
                    st.stop()
                df = st.session_state.df.copy()
                feature_cols = st.session_state.feature_cols
                num_lags = st.session_state.num_lags
                for var in st.session_state.input_vars:
                    if st.session_state.var_types[var] == "Dynamic":
                        for lag in range(1, num_lags + 1):
                            df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                    else:
                        df[var] = df[var].fillna(df[var].median())
                for lag in range(1, num_lags + 1):
                    df[f'{st.session_state.output_var}_Lag_{lag}'] = df[st.session_state.output_var].shift(lag)
                
                df = df.dropna()
                train_size = int(len(df) * train_split)
                train_df, test_df = df[:train_size], df[train_size:]
                scaler = st.session_state.scaler
                
                train_scaled = scaler.transform(train_df[feature_cols + [st.session_state.output_var]])
                test_scaled = scaler.transform(test_df[feature_cols + [st.session_state.output_var]])
                X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
                X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                layers = (st.session_state.gru_layers if model_type in ["GRU", "Hybrid"] else 
                          st.session_state.lstm_layers if model_type == "LSTM" else 
                          st.session_state.rnn_layers)
                units = (st.session_state.gru_units if model_type in ["GRU", "Hybrid"] else 
                         st.session_state.lstm_units if model_type == "LSTM" else 
                         st.session_state.rnn_units)
                st.session_state.model = build_model(
                    (X_train.shape[1], X_train.shape[2]), model_type, layers, units, 
                    st.session_state.dense_layers, st.session_state.dense_units, st.session_state.learning_rate
                )
                st.session_state.model.load_weights(MODEL_WEIGHTS_PATH)
                y_train_pred = st.session_state.model.predict(X_train)
                y_test_pred = st.session_state.model.predict(X_test)
                y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
                y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
                y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
                y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]
                
                metrics = {metric: {
                    "Training": all_metrics_dict[metric](y_train_actual, y_train_pred),
                    "Testing": all_metrics_dict[metric](y_test_actual, y_test_pred)
                } for metric in st.session_state.selected_metrics}
                st.session_state.metrics = metrics

                dates = df[st.session_state.date_col] if st.session_state.date_col else pd.RangeIndex(len(df))
                train_dates, test_dates = dates[:train_size], dates[train_size:]
                st.session_state.train_results_df = pd.DataFrame({
                    "Date": train_dates[:len(y_train_actual)],
                    f"Actual_{st.session_state.output_var}": y_train_actual,
                    f"Predicted_{st.session_state.output_var}": y_train_pred
                })
                st.session_state.test_results_df = pd.DataFrame({
                    "Date": test_dates[:len(y_test_actual)],
                    f"Actual_{st.session_state.output_var}": y_test_actual,
                    f"Predicted_{st.session_state.output_var}": y_test_pred
                })

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_dates[:len(y_train_actual)], y=y_train_actual, name="Train Actual", line=dict(color="#1f77b4")))
                fig.add_trace(go.Scatter(x=train_dates[:len(y_train_pred)], y=y_train_pred, name="Train Predicted", line=dict(color="#ff7f0e", dash="dash")))
                fig.add_trace(go.Scatter(x=test_dates[:len(y_test_actual)], y=y_test_actual, name="Test Actual", line=dict(color="#2ca02c")))
                fig.add_trace(go.Scatter(x=test_dates[:len(y_test_pred)], y=y_test_pred, name="Test Predicted", line=dict(color="#d62728", dash="dash")))
                fig.update_layout(title=f"Training and Testing: {st.session_state.output_var}", xaxis_title="Date", yaxis_title=st.session_state.output_var)
                st.session_state.fig = fig
                st.success("Model tested successfully!")

# Results Section
if any([st.session_state.metrics, st.session_state.fig, st.session_state.train_results_df, st.session_state.test_results_df]):
    with st.expander("📊 Results", expanded=True):
        st.subheader("Results Overview", divider="blue")
        if st.session_state.metrics:
            st.markdown("**📏 Performance Metrics**")
            metrics_df = pd.DataFrame({
                "Metric": st.session_state.selected_metrics,
                "Training": [f"{st.session_state.metrics[m]['Training']:.4f}" for m in st.session_state.selected_metrics],
                "Testing": [f"{st.session_state.metrics[m]['Testing']:.4f}" for m in st.session_state.selected_metrics]
            })
            st.dataframe(metrics_df.style.set_properties(**{'text-align': 'center'}), use_container_width=True)
        
        if st.session_state.fig:
            st.markdown("**📈 Prediction Plot**")
            st.plotly_chart(st.session_state.fig, use_container_width=True)
            buf = BytesIO()
            try:
                st.session_state.fig.write_image(buf, format="png")
            except ValueError:
                fig, ax = plt.subplots()
                ax.plot(st.session_state.train_results_df["Date"], st.session_state.train_results_df[f"Actual_{st.session_state.output_var}"], label="Train Actual")
                ax.plot(st.session_state.train_results_df["Date"], st.session_state.train_results_df[f"Predicted_{st.session_state.output_var}"], label="Train Predicted", linestyle="--")
                ax.plot(st.session_state.test_results_df["Date"], st.session_state.test_results_df[f"Actual_{st.session_state.output_var}"], label="Test Actual")
                ax.plot(st.session_state.test_results_df["Date"], st.session_state.test_results_df[f"Predicted_{st.session_state.output_var}"], label="Test Predicted", linestyle="--")
                ax.legend()
                ax.set_title(f"Training and Testing: {st.session_state.output_var}")
                fig.savefig(buf, format="png", bbox_inches="tight")
            st.download_button("⬇️ Download Plot", buf.getvalue(), "prediction_plot.png", "image/png", key="plot_dl")
        
        if st.session_state.train_results_df is not None:
            train_csv = st.session_state.train_results_df.to_csv(index=False)
            st.download_button("⬇️ Download Train Data CSV", train_csv, "train_predictions.csv", "text/csv", key="train_dl")
        if st.session_state.test_results_df is not None:
            test_csv = st.session_state.test_results_df.to_csv(index=False)
            st.download_button("⬇️ Download Test Data CSV", test_csv, "test_predictions.csv", "text/csv", key="test_dl")

# New Data Prediction Section
if os.path.exists(MODEL_WEIGHTS_PATH):
    with st.expander("🔮 New Predictions", expanded=False):
        st.subheader("Predict New Data", divider="blue")
        new_data_files = st.file_uploader("Upload New Data (Excel)", type=["xlsx"], accept_multiple_files=True, key="new_data")
        
        if new_data_files:
            for new_data_file in new_data_files:
                new_df = pd.read_excel(new_data_file)
                st.markdown(f"**Preview for {new_data_file.name}:**")
                st.dataframe(new_df.head(), use_container_width=True)
                
                datetime_cols = [col for col in new_df.columns if pd.api.types.is_datetime64_any_dtype(new_df[col]) or "date" in col.lower()]
                date_col = st.selectbox(f"Select Date Column ({new_data_file.name})", ["None"] + datetime_cols, index=0, 
                                        key=f"date_col_new_{new_data_file.name}")
                if date_col != "None":
                    new_df[date_col] = pd.to_datetime(new_df[date_col])
                    new_df = new_df.sort_values(date_col)
                
                input_vars = st.session_state.input_vars
                output_var = st.session_state.output_var
                num_lags = st.session_state.num_lags
                feature_cols = st.session_state.feature_cols
                
                available_new_inputs = [col for col in new_df.columns if col in input_vars and col != date_col]
                if not available_new_inputs:
                    st.error(f"No recognized input variables in {new_data_file.name}. Expected: {', '.join(input_vars)}")
                    continue
                selected_inputs = st.multiselect(f"🔧 Input Variables ({new_data_file.name})", available_new_inputs, 
                                                default=available_new_inputs, key=f"new_input_vars_{new_data_file.name}")
                
                if st.button(f"🔍 Predict ({new_data_file.name})", key=f"predict_button_{new_data_file.name}"):
                    if len(new_df) < num_lags + 1:
                        st.error(f"{new_data_file.name} has insufficient rows ({len(new_df)}) for {num_lags} lags.")
                        continue
                    
                    new_df_processed = new_df.copy()
                    feature_cols_new = []
                    for var in selected_inputs:
                        if st.session_state.var_types[var] == "Dynamic":
                            for lag in range(1, num_lags + 1):
                                new_df_processed[f'{var}_Lag_{lag}'] = new_df_processed[var].shift(lag)
                                feature_cols_new.append(f'{var}_Lag_{lag}')
                        else:
                            feature_cols_new.append(var)
                    for lag in range(1, num_lags + 1):
                        new_df_processed[f'{output_var}_Lag_{lag}'] = new_df_processed[output_var].shift(lag) if output_var in new_df_processed.columns else np.nan
                        feature_cols_new.append(f'{output_var}_Lag_{lag}')
                    
                    full_new_df = pd.DataFrame(index=new_df_processed.index, columns=feature_cols + [output_var])
                    for col in feature_cols:
                        if col in new_df_processed.columns:
                            full_new_df[col] = new_df_processed[col]
                        else:
                            full_new_df[col] = np.nan
                    full_new_df[output_var] = new_df_processed[output_var] if output_var in new_df_processed.columns else 0
                    full_new_df = full_new_df.fillna(full_new_df.median(numeric_only=True)).fillna(0)
                    
                    scaler = st.session_state.scaler
                    new_scaled = scaler.transform(full_new_df[feature_cols + [output_var]])
                    X_new = new_scaled[:, :-1]
                    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
                    
                    layers = (st.session_state.gru_layers if model_type in ["GRU", "Hybrid"] else 
                              st.session_state.lstm_layers if model_type == "LSTM" else 
                              st.session_state.rnn_layers)
                    units = (st.session_state.gru_units if model_type in ["GRU", "Hybrid"] else 
                             st.session_state.lstm_units if model_type == "LSTM" else 
                             st.session_state.rnn_units)
                    if st.session_state.model is None:
                        st.session_state.model = build_model(
                            (X_new.shape[1], X_new.shape[2]), model_type, layers, units, 
                            st.session_state.dense_layers, st.session_state.dense_units, st.session_state.learning_rate
                        )
                        st.session_state.model.load_weights(MODEL_WEIGHTS_PATH)
                    y_new_pred = st.session_state.model.predict(X_new)
                    y_new_pred = scaler.inverse_transform(np.hstack([y_new_pred, X_new[:, 0, :]]))[:, 0]
                    y_new_pred = np.maximum(y_new_pred, 0)  # Ensure non-negative
                    
                    dates = new_df[date_col] if date_col != "None" else pd.RangeIndex(len(new_df))
                    valid_indices = ~new_df_processed[feature_cols].isna().any(axis=1)
                    valid_dates = dates[valid_indices][-len(y_new_pred):]
                    new_predictions_df = pd.DataFrame({
                        "Date": valid_dates,
                        f"Predicted_{output_var}": y_new_pred
                    })
                    st.session_state[f"new_predictions_df_{new_data_file.name}"] = new_predictions_df
                    
                    st.write(f"Debug: Length of dates: {len(dates)}, Predictions: {len(y_new_pred)}, Valid dates: {len(valid_dates)}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=valid_dates, y=y_new_pred, name="Predicted", mode='lines', line=dict(color="#ff7f0e")))
                    if output_var in new_df.columns:
                        fig.add_trace(go.Scatter(x=valid_dates, y=new_df[output_var][valid_indices][-len(y_new_pred):], name="Actual", mode='lines', line=dict(color="#1f77b4")))
                    fig.update_layout(title=f"New Predictions: {output_var} ({new_data_file.name})", xaxis_title="Date", yaxis_title=output_var)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    buf = BytesIO()
                    try:
                        fig.write_image(buf, format="png")
                    except ValueError:
                        fig_alt, ax = plt.subplots()
                        ax.plot(valid_dates, y=y_new_pred, label="Predicted")
                        if output_var in new_df.columns:
                            ax.plot(valid_dates, y=new_df[output_var][valid_indices][-len(y_new_pred):], label="Actual")
                        ax.legend()
                        ax.set_title(f"New Predictions: {output_var}")
                        fig_alt.savefig(buf, format="png", bbox_inches="tight")
                    st.download_button(f"⬇️ Download Plot ({new_data_file.name})", buf.getvalue(), f"new_prediction_{new_data_file.name}.png", "image/png")
                    new_csv = new_predictions_df.to_csv(index=False)
                    st.download_button(f"⬇️ Download CSV ({new_data_file.name})", new_csv, f"new_predictions_{new_data_file.name}.csv", "text/csv")
                    st.success(f"Predictions for {new_data_file.name} generated successfully!")
