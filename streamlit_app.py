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
DEFAULT_DENSE_UNITS = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_SPLIT = 80
DEFAULT_NUM_LAGS = 3
MODEL_WEIGHTS_PATH = os.path.join(tempfile.gettempdir(), "gru_model_weights.weights.h5")
MODEL_FULL_PATH = os.path.join(tempfile.gettempdir(), "gru_model.h5")
MODEL_PLOT_PATH = os.path.join(tempfile.gettempdir(), "gru_model_plot.png")

# -------------------- Metric Functions --------------------
def nse(actual, predicted):
    return 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

def kge(actual, predicted):
    r = np.corrcoef(actual, predicted)[0, 1]
    alpha = np.std(predicted) / np.std(actual)
    beta = np.mean(predicted) / np.mean(actual)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def pbias(actual, predicted):
    return 100 * (np.sum(predicted - actual) / np.sum(actual))

def peak_flow_error(actual, predicted):
    actual_peak = np.max(actual)
    predicted_peak = np.max(predicted)
    return (predicted_peak - actual_peak) / actual_peak * 100 if actual_peak != 0 else 0

def high_flow_bias(actual, predicted, percentile=90):
    threshold = np.percentile(actual, percentile)
    high_actual = actual[actual >= threshold]
    high_predicted = predicted[actual >= threshold]
    return 100 * (np.mean(high_predicted) - np.mean(high_actual)) / np.mean(high_actual) if len(high_actual) > 0 else 0

def low_flow_bias(actual, predicted, percentile=10):
    threshold = np.percentile(actual, percentile)
    low_actual = actual[actual <= threshold]
    low_predicted = predicted[actual <= threshold]
    return 100 * (np.mean(low_predicted) - np.mean(low_actual)) / np.mean(low_actual) if len(low_actual) > 0 else 0

def volume_error(actual, predicted):
    return 100 * (np.sum(predicted) - np.sum(actual)) / np.sum(actual)

# Global Metrics Dictionary
all_metrics_dict = {
    "RMSE": lambda a, p: np.sqrt(mean_squared_error(a, p)),
    "MAE": lambda a, p: mean_absolute_error(a, p),
    "R²": lambda a, p: r2_score(a, p),
    "NSE": nse,
    "KGE": kge,
    "PBIAS": pbias,
    "Peak Flow Error": peak_flow_error,
    "High Flow Bias": high_flow_bias,
    "Low Flow Bias": low_flow_bias,
    "Volume Error": volume_error
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

# -------------------- GRU Model Definition --------------------
@st.cache_resource
def build_gru_model(input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate):
    model = tf.keras.Sequential()
    for i in range(gru_layers):
        if i == 0:
            model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=(i < gru_layers - 1), input_shape=input_shape))
        else:
            model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=(i < gru_layers - 1)))
        model.add(tf.keras.layers.Dropout(0.2))
    for units in dense_units[:dense_layers]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# -------------------- Hyperparameter Suggestion --------------------
def suggest_hyperparams(X_train, y_train):
    best_config = None
    best_loss = float('inf')
    for units in [32, 64, 128]:
        model = build_gru_model((X_train.shape[1], X_train.shape[2]), 1, 1, [units], [32], 0.001)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        loss = model.evaluate(X_train, y_train, verbose=0)
        if loss < best_loss:
            best_loss = loss
            best_config = units
    return best_config

# -------------------- Styling and Streamlit UI --------------------
st.set_page_config(page_title="Wateran", page_icon="🌊", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .metric-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌊 Wateran: Time Series Prediction with GRU")
st.markdown("**Simple, Fast, and Accurate Predictions Powered by GRU Neural Networks**", unsafe_allow_html=True)

# Initialize session state
for key in ['metrics', 'train_results_df', 'test_results_df', 'fig', 'model_plot', 'scaler', 'input_vars', 'output_var', 
            'new_predictions_df', 'new_fig', 'gru_layers', 'dense_layers', 'gru_units', 'dense_units', 'learning_rate', 
            'feature_cols', 'new_data_file', 'selected_inputs', 'new_date_col', 'selected_metrics', 'var_types', 
            'new_var_types', 'num_lags', 'date_col', 'df', 'cv_metrics', 'X_train', 'y_train', 'X_test', 'y_test', 'model']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'num_lags' else DEFAULT_NUM_LAGS

# Sidebar for Navigation and Help
with st.sidebar:
    st.header("Navigation")
    st.button("📥 Data Input", key="nav_data")
    st.button("⚙️ Model Configuration", key="nav_config")
    st.button("📊 Results", key="nav_results")
    st.button("🔮 New Predictions", key="nav_predict")
    with st.expander("ℹ️ Help"):
        st.markdown("""
        - **GRU Layers**: Recurrent layers for time dependencies (1-5 recommended).
        - **Dense Layers**: Fully connected layers for output refinement.
        - **Dynamic Variables**: Use lagged values for time series modeling.
        - **Static Variables**: Constant features, no lags applied.
        - **Metrics**: NSE/KGE ideal = 1, PBIAS ideal = 0.
        """)

# Main Layout
col1, col2 = st.columns([2, 1], gap="large")

# Left Column: Data and Variable Selection
with col1:
    st.subheader("📥 Data Input", divider="blue")
    uploaded_file = st.file_uploader("Upload Training Data (Excel)", type=["xlsx"], key="train_data", help="Excel file with time series data (e.g., dates, flows).")
    
    if uploaded_file:
        @st.cache_data
        def load_data(file):
            return pd.read_excel(file)
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.markdown("**Dataset Preview:**")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Date column selection
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        date_col = st.selectbox("Select Date Column (optional)", ["None"] + datetime_cols, index=0, key="date_col_train", help="Choose a date column or use index.")
        st.session_state.date_col = date_col if date_col != "None" else None
        if date_col != "None":
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        
        # Numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != st.session_state.date_col]
        if len(numeric_cols) < 2:
            st.error("Dataset requires at least two numeric columns.")
            st.stop()
        
        # Variable selection
        st.markdown("**Variable Selection**")
        output_var = st.selectbox("🎯 Output Variable", numeric_cols, key="output_var_train", help="Variable to predict (e.g., streamflow).")
        available_input_cols = [col for col in numeric_cols if col != output_var]
        default_input = [available_input_cols[0]] if available_input_cols else []
        input_vars = st.multiselect(
            "🔧 Input Variables",
            available_input_cols,
            default=default_input,
            key="input_vars_train",
            help="Features for prediction (e.g., rainfall)."
        )
        if not input_vars:
            st.error("Select at least one input variable.")
            st.stop()

        # Variable types
        with st.expander("Variable Types", expanded=True):
            var_types = {}
            for var in input_vars:
                var_types[var] = st.selectbox(f"{var} Type", ["Dynamic", "Static"], key=f"{var}_type", help="Dynamic: Lagged; Static: Constant.")
            st.session_state.var_types = var_types
        
        # Data Exploration
        with st.expander("📋 Data Exploration"):
            st.markdown("**Summary Statistics**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            fig, ax = plt.subplots()
            df[numeric_cols].plot(ax=ax)
            st.pyplot(fig)
            st.markdown("**Correlation Heatmap**")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.session_state.input_vars = input_vars
        st.session_state.output_var = output_var

# Right Column: Model Settings and Actions
with col2:
    st.subheader("⚙️ Model Configuration", divider="blue")
    
    # Load Saved Model
    uploaded_model = st.file_uploader("Load Saved Model", type=["h5"], help="Upload a previously saved GRU model.")
    if uploaded_model:
        st.session_state.model = tf.keras.models.load_model(uploaded_model)
        st.success("Model loaded successfully!")
    
    # Training Parameters
    st.markdown("**Training Parameters**")
    epochs = st.slider("Epochs", 1, 1500, DEFAULT_EPOCHS, step=10, help="Number of training iterations (more = longer training).")
    batch_size = st.slider("Batch Size", 8, 128, DEFAULT_BATCH_SIZE, step=8, help="Samples per gradient update.")
    train_split = st.slider("Training Data %", 50, 90, DEFAULT_TRAIN_SPLIT, help="Percentage for training vs. testing.") / 100
    st.number_input("Number of Lags", min_value=1, max_value=10, value=DEFAULT_NUM_LAGS if st.session_state.num_lags is None else st.session_state.num_lags, step=1, key="num_lags", help="Past time steps to include.")
    
    # Model Architecture
    with st.expander("Model Architecture", expanded=False):
        gru_layers = st.number_input("GRU Layers", min_value=1, max_value=5, value=1, step=1, help="Number of GRU layers for time modeling.")
        gru_units = [st.number_input(f"GRU Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") for i in range(gru_layers)]
        dense_layers = st.number_input("Dense Layers", min_value=1, max_value=5, value=1, step=1, help="Number of dense layers for output.")
        dense_units = [st.number_input(f"Dense Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") for i in range(dense_layers)]
        learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=DEFAULT_LEARNING_RATE, format="%.5f", help="Optimization step size.")
        st.session_state.gru_layers = gru_layers
        st.session_state.dense_layers = dense_layers
        st.session_state.gru_units = gru_units
        st.session_state.dense_units = dense_units
        st.session_state.learning_rate = learning_rate
    
    # Metrics Selection
    st.markdown("**Evaluation Metrics**")
    all_metrics = ["RMSE", "MAE", "R²", "NSE", "KGE", "PBIAS", "Peak Flow Error", "High Flow Bias", "Low Flow Bias", "Volume Error"]
    selected_metrics = st.multiselect("Select Metrics", all_metrics, default=all_metrics, key="metrics_select", help="Metrics to evaluate performance (e.g., NSE for efficiency).")
    st.session_state.selected_metrics = selected_metrics or all_metrics

    # Training and Testing Buttons
    if uploaded_file:
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🚀 Train Model"):
                df = st.session_state.df.copy()
                selected_cols = st.session_state.input_vars + [st.session_state.output_var]
                if df[selected_cols].isnull().sum().sum() > 0:
                    st.warning("Missing values detected. Handled during preprocessing.")

                # Feature Engineering
                feature_cols = []
                num_lags = st.session_state["num_lags"]
                for var in st.session_state.input_vars:
                    if st.session_state.var_types[var] == "Dynamic":
                        for lag in range(1, num_lags + 1):
                            df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                            feature_cols.append(f'{var}_Lag_{lag}')
                    else:
                        df[var] = df[var].fillna(0) if df[var].isnull().sum() > len(df) * 0.9 else df[var].fillna(df[var].median())
                        feature_cols.append(var)
                for lag in range(1, num_lags + 1):
                    df[f'{st.session_state.output_var}_Lag_{lag}'] = df[st.session_state.output_var].shift(lag)
                    feature_cols.append(f'{st.session_state.output_var}_Lag_{lag}')
                
                df = df.dropna(subset=[col for col in feature_cols if "_Lag_" in col], how='all')
                df[feature_cols] = df[feature_cols].fillna(0)
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
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.session_state.model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
                with st.spinner("Training in progress..."):
                    progress_placeholder = st.empty()
                    callback = StreamlitProgressCallback(epochs, progress_placeholder)
                    st.session_state.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[callback, early_stopping, lr_scheduler])
                    st.session_state.model.save_weights(MODEL_WEIGHTS_PATH)
                    st.session_state.model.save(MODEL_FULL_PATH)
                st.success("Model trained and saved successfully!")
        
        with col_btn2:
            if st.button("🤖 Suggest GRU Units"):
                if "X_train" not in st.session_state or "y_train" not in st.session_state:
                    st.error("Please train the model first to generate training data.")
                else:
                    suggested_units = suggest_hyperparams(st.session_state.X_train, st.session_state.y_train)
                    st.info(f"Suggested GRU Units: {suggested_units}")
        
        with col_btn3:
            if st.button("🔍 Test Model"):
                if not os.path.exists(MODEL_WEIGHTS_PATH):
                    st.error("Train the model first!")
                    st.stop()
                df = st.session_state.df.copy()
                feature_cols = st.session_state.feature_cols
                num_lags = st.session_state["num_lags"]
                for var in st.session_state.input_vars:
                    if st.session_state.var_types[var] == "Dynamic":
                        for lag in range(1, num_lags + 1):
                            df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                    else:
                        df[var] = df[var].fillna(0) if df[var].isnull().sum() > len(df) * 0.9 else df[var].fillna(df[var].median())
                for lag in range(1, num_lags + 1):
                    df[f'{st.session_state.output_var}_Lag_{lag}'] = df[st.session_state.output_var].shift(lag)
                
                df = df.dropna(subset=[col for col in feature_cols if "_Lag_" in col], how='all')
                df[feature_cols] = df[feature_cols].fillna(0)
                train_size = int(len(df) * train_split)
                train_df, test_df = df[:train_size], df[train_size:]
                scaler = st.session_state.scaler
                
                train_scaled = scaler.transform(train_df[feature_cols + [st.session_state.output_var]])
                test_scaled = scaler.transform(test_df[feature_cols + [st.session_state.output_var]])
                X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
                X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                st.session_state.model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                st.session_state.model.load_weights(MODEL_WEIGHTS_PATH)
                y_train_pred = st.session_state.model.predict(X_train)
                y_test_pred = st.session_state.model.predict(X_test)
                y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
                y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
                y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
                y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]
                y_train_pred, y_test_pred = np.clip(y_train_pred, 0, None), np.clip(y_test_pred, 0, None)

                metrics = {metric: {
                    "Training": all_metrics_dict[metric](y_train_actual, y_train_pred),
                    "Testing": all_metrics_dict[metric](y_test_actual, y_test_pred)
                } for metric in selected_metrics}
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

                # Interactive Plotly Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_dates[:len(y_train_actual)], y=y_train_actual, name="Train Actual", line=dict(color="#1f77b4")))
                fig.add_trace(go.Scatter(x=train_dates[:len(y_train_pred)], y=y_train_pred, name="Train Predicted", line=dict(color="#ff7f0e", dash="dash")))
                fig.add_trace(go.Scatter(x=test_dates[:len(y_test_actual)], y=y_test_actual, name="Test Actual", line=dict(color="#2ca02c")))
                fig.add_trace(go.Scatter(x=test_dates[:len(y_test_pred)], y=y_test_pred, name="Test Predicted", line=dict(color="#d62728", dash="dash")))
                fig.update_layout(title=f"Training and Testing: {st.session_state.output_var}", xaxis_title="Date", yaxis_title=st.session_state.output_var)
                st.session_state.fig = fig
                st.success("Model tested successfully!")

# Cross-Validation Section
if st.session_state.feature_cols:
    with st.expander("🔄 Cross-Validation", expanded=False):
        if st.button("Run Cross-Validation"):
            df = st.session_state.df.copy()
            feature_cols = st.session_state.feature_cols
            num_lags = st.session_state["num_lags"]
            for var in st.session_state.input_vars:
                if st.session_state.var_types[var] == "Dynamic":
                    for lag in range(1, num_lags + 1):
                        df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                else:
                    df[var] = df[var].fillna(0) if df[var].isnull().sum() > len(df) * 0.9 else df[var].fillna(df[var].median())
            for lag in range(1, num_lags + 1):
                df[f'{st.session_state.output_var}_Lag_{lag}'] = df[st.session_state.output_var].shift(lag)
            
            df = df.dropna(subset=[col for col in feature_cols if "_Lag_" in col], how='all')
            df[feature_cols] = df[feature_cols].fillna(0)
            scaler = st.session_state.scaler
            scaled = scaler.transform(df[feature_cols + [st.session_state.output_var]])
            X, y = scaled[:, :-1], scaled[:, -1]
            X = X.reshape((X.shape[0], 1, X.shape[1]))

            tscv = TimeSeriesSplit(n_splits=5)
            cv_metrics = {metric: [] for metric in selected_metrics}
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                model = build_gru_model((X_tr.shape[1], X_tr.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
                y_val_pred = model.predict(X_val)
                y_val_pred = scaler.inverse_transform(np.hstack([y_val_pred, X_val[:, 0, :]]))[:, 0]
                y_val_actual = scaler.inverse_transform(np.hstack([y_val.reshape(-1, 1), X_val[:, 0, :]]))[:, 0]
                for metric in selected_metrics:
                    cv_metrics[metric].append(all_metrics_dict[metric](y_val_actual, y_val_pred))
            st.session_state.cv_metrics = {m: np.mean(cv_metrics[m]) for m in selected_metrics}
            st.write("Cross-Validation Results:", st.session_state.cv_metrics)

# Results Section
if any([st.session_state.metrics, st.session_state.fig, st.session_state.train_results_df, st.session_state.test_results_df]):
    with st.expander("📊 Results", expanded=True):
        st.subheader("Results Overview", divider="blue")
        if st.session_state.metrics:
            st.markdown("**📏 Performance Metrics**")
            metrics_df = pd.DataFrame({
                "Metric": selected_metrics,
                "Training": [f"{st.session_state.metrics[m]['Training']:.4f}" for m in selected_metrics],
                "Testing": [f"{st.session_state.metrics[m]['Testing']:.4f}" for m in selected_metrics]
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
                fig.savefig(buf, format="png", bbox_inches="tight")
            st.download_button("⬇️ Download Plot", buf.getvalue(), "prediction_plot.png", "image/png", key="plot_dl")
        
        if st.session_state.train_results_df is not None:
            train_csv = st.session_state.train_results_df.to_csv(index=False)
            st.download_button("⬇️ Train Data CSV", train_csv, "train_predictions.csv", "text/csv", key="train_dl")
        if st.session_state.test_results_df is not None:
            test_csv = st.session_state.test_results_df.to_csv(index=False)
            st.download_button("⬇️ Test Data CSV", test_csv, "test_predictions.csv", "text/csv", key="test_dl")
        
        if st.button("🗺️ Show Model Architecture"):
            if st.session_state.model is None:
                st.error("No model available. Please train, test, or load a model first.")
            else:
                plot_model(st.session_state.model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True)
                st.image(MODEL_PLOT_PATH, caption="Model Architecture")

# New Data Prediction Section
if os.path.exists(MODEL_WEIGHTS_PATH):
    with st.expander("🔮 New Predictions", expanded=False):
        st.subheader("Predict New Data", divider="blue")
        new_data_files = st.file_uploader("Upload New Data (Excel)", type=["xlsx"], accept_multiple_files=True, key="new_data", help="Upload one or more Excel files for prediction.")
        
        if new_data_files:
            for new_data_file in new_data_files:
                new_df = pd.read_excel(new_data_file)
                st.markdown(f"**Preview for {new_data_file.name}:**")
                st.dataframe(new_df.head(), use_container_width=True)
                
                datetime_cols = [col for col in new_df.columns if pd.api.types.is_datetime64_any_dtype(new_df[col]) or "date" in col.lower()]
                date_col = st.selectbox(f"Select Date Column ({new_data_file.name})", ["None"] + datetime_cols, index=0, key=f"date_col_new_{new_data_file.name}")
                if date_col != "None":
                    new_df[date_col] = pd.to_datetime(new_df[date_col])
                    new_df = new_df.sort_values(date_col)
                
                input_vars = st.session_state.input_vars
                output_var = st.session_state.output_var
                num_lags = st.session_state["num_lags"]
                feature_cols = st.session_state.feature_cols
                
                available_new_inputs = [col for col in new_df.columns if col in input_vars and col != date_col]
                if not available_new_inputs:
                    st.error(f"No recognized input variables in {new_data_file.name}. Include: " + ", ".join(input_vars))
                    continue
                selected_inputs = st.multiselect(f"🔧 Input Variables ({new_data_file.name})", available_new_inputs, default=available_new_inputs, key=f"new_input_vars_{new_data_file.name}")
                
                st.markdown(f"**Variable Types ({new_data_file.name})**")
                new_var_types = {}
                for var in selected_inputs:
                    new_var_types[var] = st.selectbox(f"{var} Type", ["Dynamic", "Static"], key=f"new_{var}_type_{new_data_file.name}")
                
                if st.button(f"🔍 Predict ({new_data_file.name})"):
                    if len(new_df) < (num_lags + 1 if any(new_var_types[var] == "Dynamic" for var in selected_inputs) else 1):
                        st.error(f"{new_data_file.name} has insufficient rows for {num_lags} lags.")
                        continue
                    
                    feature_cols_new = []
                    for var in selected_inputs:
                        if new_var_types[var] == "Dynamic":
                            for lag in range(1, num_lags + 1):
                                new_df[f'{var}_Lag_{lag}'] = new_df[var].shift(lag)
                                feature_cols_new.append(f'{var}_Lag_{lag}')
                        else:
                            feature_cols_new.append(var)
                    for lag in range(1, num_lags + 1):
                        new_df[f'{output_var}_Lag_{lag}'] = new_df[output_var].shift(lag) if output_var in new_df.columns else 0
                        feature_cols_new.append(f'{output_var}_Lag_{lag}')
                    
                    new_df.dropna(subset=[col for col in feature_cols_new if "_Lag_" in col], how='all', inplace=True)
                    full_new_df = pd.DataFrame(index=new_df.index, columns=feature_cols + [output_var])
                    full_new_df[output_var] = new_df[output_var] if output_var in new_df.columns else 0
                    for col in feature_cols_new:
                        if col in full_new_df.columns and col in new_df.columns:
                            full_new_df[col] = new_df[col]
                    full_new_df.fillna(0, inplace=True)
                    full_new_df = full_new_df[feature_cols + [output_var]].apply(pd.to_numeric, errors='coerce')
                    
                    scaler = st.session_state.scaler
                    new_scaled = scaler.transform(full_new_df[feature_cols + [output_var]])
                    X_new = new_scaled[:, :-1]
                    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
                    
                    if st.session_state.model is None:
                        st.session_state.model = build_gru_model((X_new.shape[1], X_new.shape[2]), st.session_state.gru_layers, st.session_state.dense_layers, st.session_state.gru_units, st.session_state.dense_units, st.session_state.learning_rate)
                        st.session_state.model.load_weights(MODEL_WEIGHTS_PATH)
                    y_new_pred = st.session_state.model.predict(X_new)
                    y_new_pred = scaler.inverse_transform(np.hstack([y_new_pred, X_new[:, 0, :]]))[:, 0]
                    y_new_pred = np.clip(y_new_pred, 0, None)
                    
                    dates = new_df[date_col] if date_col != "None" else pd.RangeIndex(len(new_df))
                    new_predictions_df = pd.DataFrame({
                        "Date": dates.values[-len(y_new_pred):],
                        f"Predicted_{output_var}": y_new_pred
                    })
                    st.session_state[f"new_predictions_df_{new_data_file.name}"] = new_predictions_df
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates.values[-len(y_new_pred):], y=y_new_pred, name="Predicted", line=dict(color="#ff7f0e")))
                    fig.update_layout(title=f"New Predictions: {output_var} ({new_data_file.name})", xaxis_title="Date", yaxis_title=output_var)
                    st.plotly_chart(fig, use_container_width=True)
                    buf = BytesIO()
                    try:
                        fig.write_image(buf, format="png")
                    except ValueError:
                        fig_alt, ax = plt.subplots()
                        ax.plot(dates.values[-len(y_new_pred):], y_new_pred, label="Predicted")
                        ax.legend()
                        fig_alt.savefig(buf, format="png", bbox_inches="tight")
                    st.download_button(f"⬇️ Download Plot ({new_data_file.name})", buf.getvalue(), f"new_prediction_{new_data_file.name}.png", "image/png")
                    new_csv = new_predictions_df.to_csv(index=False)
                    st.download_button(f"⬇️ Download CSV ({new_data_file.name})", new_csv, f"new_predictions_{new_data_file.name}.csv", "text/csv")
                    st.success(f"Predictions for {new_data_file.name} generated successfully!")
