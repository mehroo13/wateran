import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

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
    if len(high_actual) > 0:
        return 100 * (np.mean(high_predicted) - np.mean(high_actual)) / np.mean(high_actual)
    return 0

def low_flow_bias(actual, predicted, percentile=10):
    threshold = np.percentile(actual, percentile)
    low_actual = actual[actual <= threshold]
    low_predicted = predicted[actual <= threshold]
    if len(low_actual) > 0:
        return 100 * (np.mean(low_predicted) - np.mean(low_actual)) / np.mean(low_actual)
    return 0

def volume_error(actual, predicted):
    return 100 * (np.sum(predicted) - np.sum(actual)) / np.sum(actual)

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

# -------------------- GRU Model Definition --------------------
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

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Wateran", page_icon="🌊", layout="wide")
st.title("🌊 Wateran")
st.markdown("**Predict time series data with GRU - Simple, Fast, Accurate!**")

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
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'input_vars' not in st.session_state:
    st.session_state.input_vars = []
if 'output_var' not in st.session_state:
    st.session_state.output_var = None
if 'new_predictions_df' not in st.session_state:
    st.session_state.new_predictions_df = None
if 'new_fig' not in st.session_state:
    st.session_state.new_fig = None
if 'gru_layers' not in st.session_state:
    st.session_state.gru_layers = None
if 'dense_layers' not in st.session_state:
    st.session_state.dense_layers = None
if 'gru_units' not in st.session_state:
    st.session_state.gru_units = None
if 'dense_units' not in st.session_state:
    st.session_state.dense_units = None
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'new_data_file' not in st.session_state:
    st.session_state.new_data_file = None
if 'selected_inputs' not in st.session_state:
    st.session_state.selected_inputs = []
if 'new_date_col' not in st.session_state:
    st.session_state.new_date_col = None
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = None

# Main layout with two columns
col1, col2 = st.columns([2, 1])

# Left Column: Data and Variable Selection
with col1:
    st.subheader("📥 Data Input")
    uploaded_file = st.file_uploader("Upload Training Data (Excel)", type=["xlsx"], key="train_data")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("**Dataset Preview:**", df.head(5))
        
        # Date column selection
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        date_col = None
        if datetime_cols:
            date_col = st.selectbox("Select Date Column (optional)", ["None"] + datetime_cols, index=0, key="date_col_train")
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            else:
                st.info("Using index for ordering.")
        else:
            st.info("No datetime column detected. Using index.")
        
        # Numeric columns for variable selection
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (date_col is None or col != date_col)]
        if len(numeric_cols) < 2:
            st.error("Dataset requires at least one input and one output variable (numeric columns).")
            st.stop()
        
        # Variable selection
        output_var = st.selectbox("🎯 Output Variable", numeric_cols, key="output_var_train")
        available_input_cols = [col for col in numeric_cols if col != output_var]
        if not available_input_cols:
            st.error("No input variables available. Select a different output variable or check your dataset.")
            st.stop()
        input_vars = st.multiselect("🔧 Input Variables", available_input_cols, default=available_input_cols[:1] if available_input_cols else [], key="input_vars")
        if not input_vars:
            st.warning("No input variables selected. At least one is recommended for analysis.")

        # Generate lagged features for all input variables
        feature_cols = []
        for var in input_vars + [output_var]:
            for lag in range(1, NUM_LAGGED_FEATURES + 1):
                df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                feature_cols.append(f'{var}_Lag_{lag}')
        df.dropna(inplace=True)

        # Store in session state
        st.session_state.input_vars = input_vars
        st.session_state.output_var = output_var
        st.session_state.feature_cols = feature_cols
        st.session_state.date_col = date_col

# Right Column: Model Settings and Actions
with col2:
    st.subheader("⚙️ Model Configuration")
    
    # Training Parameters (always visible)
    epochs = st.slider("Epochs", 1, 1500, DEFAULT_EPOCHS, step=10)
    batch_size = st.slider("Batch Size", 8, 128, DEFAULT_BATCH_SIZE, step=8)
    train_split = st.slider("Training Data %", 50, 90, DEFAULT_TRAIN_SPLIT) / 100
    
    # Model Architecture (always visible)
    with st.expander("Model Architecture", expanded=False):
        gru_layers = st.number_input("GRU Layers", min_value=1, max_value=5, value=1, step=1)
        gru_units = [st.number_input(f"GRU Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") for i in range(gru_layers)]
        dense_layers = st.number_input("Dense Layers", min_value=1, max_value=5, value=1, step=1)
        dense_units = [st.number_input(f"Dense Layer {i+1} Units", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") for i in range(dense_layers)]
        learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.1, value=DEFAULT_LEARNING_RATE, format="%.5f")
    
    # Metrics Selection (always visible)
    st.markdown("**Evaluation Metrics**")
    all_metrics = ["RMSE", "MAE", "R²", "NSE", "KGE", "PBIAS", "Peak Flow Error", "High Flow Bias", "Low Flow Bias", "Volume Error"]
    if st.session_state.selected_metrics is None:
        st.session_state.selected_metrics = all_metrics
    selected_metrics = st.multiselect("Select Metrics", all_metrics, default=st.session_state.selected_metrics, key="metrics_select")
    st.session_state.selected_metrics = selected_metrics
    if not selected_metrics:
        st.error("Please select at least one metric.")
        st.stop()

    # Store model settings in session state
    st.session_state.gru_layers = gru_layers
    st.session_state.dense_layers = dense_layers
    st.session_state.gru_units = gru_units
    st.session_state.dense_units = dense_units
    st.session_state.learning_rate = learning_rate

    # Training and Testing Buttons (visible only with data)
    if uploaded_file:
        st.subheader("📈 Actions")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Train Model"):
                train_size = int(len(df) * train_split)
                train_df, test_df = df[:train_size], df[train_size:]
                all_input_cols = input_vars + [output_var]
                all_feature_cols = feature_cols
                
                scaler = MinMaxScaler()
                train_scaled = scaler.fit_transform(train_df[all_input_cols + all_feature_cols + [output_var]])
                test_scaled = scaler.transform(test_df[all_input_cols + all_feature_cols + [output_var]])
                st.session_state.scaler = scaler
                
                X_train = train_scaled[:, 1:1 + len(feature_cols)]
                y_train = train_scaled[:, 0]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                
                X_test = test_scaled[:, 1:1 + len(feature_cols)]
                y_test = test_scaled[:, 0]
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                # Add EarlyStopping to prevent overfitting
                early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                try:
                    with st.spinner("Training in progress..."):
                        progress_placeholder = st.empty()
                        callback = StreamlitProgressCallback(epochs, progress_placeholder)
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[callback, early_stopping])
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
                train_size = int(len(df) * train_split)
                train_df, test_df = df[:train_size], df[train_size:]
                all_input_cols = input_vars + [output_var]
                all_feature_cols = feature_cols
                
                scaler = st.session_state.scaler
                train_scaled = scaler.transform(train_df[all_input_cols + all_feature_cols + [output_var]])
                test_scaled = scaler.transform(test_df[all_input_cols + all_feature_cols + [output_var]])
                
                X_train = train_scaled[:, 1:1 + len(feature_cols)]
                y_train = train_scaled[:, 0]
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                
                X_test = test_scaled[:, 1:1 + len(feature_cols)]
                y_test = test_scaled[:, 0]
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
                try:
                    model.load_weights(MODEL_WEIGHTS_PATH)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    # Inverse transform using the full scaled data structure
                    train_full = np.zeros((len(y_train), len(all_input_cols) + len(all_feature_cols) + 1))
                    train_full[:, 0] = y_train
                    train_full[:, 1:1 + len(feature_cols)] = X_train[:, 0, :]
                    test_full = np.zeros((len(y_test), len(all_input_cols) + len(all_feature_cols) + 1))
                    test_full[:, 0] = y_test
                    test_full[:, 1:1 + len(feature_cols)] = X_test[:, 0, :]
                    
                    y_train_pred_full = np.zeros_like(train_full)
                    y_train_pred_full[:, 0] = y_train_pred.flatten()
                    y_train_pred_full[:, 1:] = train_full[:, 1:]
                    y_train_pred_inv = scaler.inverse_transform(y_train_pred_full)[:, 0]
                    
                    y_test_pred_full = np.zeros_like(test_full)
                    y_test_pred_full[:, 0] = y_test_pred.flatten()
                    y_test_pred_full[:, 1:] = test_full[:, 1:]
                    y_test_pred_inv = scaler.inverse_transform(y_test_pred_full)[:, 0]
                    
                    y_train_actual = scaler.inverse_transform(train_full)[:, 0]
                    y_test_actual = scaler.inverse_transform(test_full)[:, 0]
                    
                    y_train_pred = np.clip(y_train_pred_inv, 0, None)
                    y_test_pred = np.clip(y_test_pred_inv, 0, None)

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

                    metrics = {metric: {
                        "Training": all_metrics_dict[metric](y_train_actual, y_train_pred),
                        "Testing": all_metrics_dict[metric](y_test_actual, y_test_pred)
                    } for metric in selected_metrics}
                    st.session_state.metrics = metrics

                    dates = df[date_col] if date_col != "None" else pd.RangeIndex(len(df))
                    train_dates, test_dates = dates[:train_size], dates[train_size:]
                    st.session_state.train_results_df = pd.DataFrame({
                        "Date": train_dates[:len(y_train_actual)],
                        f"Actual_{output_var}": y_train_actual,
                        f"Predicted_{output_var}": y_train_pred
                    })
                    st.session_state.test_results_df = pd.DataFrame({
                        "Date": test_dates[:len(y_test_actual)],
                        f"Actual_{output_var}": y_test_actual,
                        f"Predicted_{output_var}": y_test_pred
                    })
                    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                    ax[0].plot(train_dates[:len(y_train_actual)], y_train_actual, label="Actual", color="#1f77b4", linewidth=2)
                    ax[0].plot(train_dates[:len(y_train_pred)], y_train_pred, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
                    ax[0].set_title(f"Training Data: {output_var}", fontsize=14, pad=10)
                    ax[0].legend()
                    ax[0].grid(True, linestyle='--', alpha=0.7)
                    if date_col != "None":
                        ax[0].set_xlabel("Date")
                        plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
                    ax[1].plot(test_dates[:len(y_test_actual)], y_test_actual, label="Actual", color="#1f77b4", linewidth=2)
                    ax[1].plot(test_dates[:len(y_test_pred)], y_test_pred, label="Predicted", color="#ff7f0e", linestyle="--", linewidth=2)
                    ax[1].set_title(f"Testing Data: {output_var}", fontsize=14, pad=10)
                    ax[1].legend()
                    ax[1].grid(True, linestyle='--', alpha=0.7)
                    if date_col != "None":
                        ax[1].set_xlabel("Date")
                        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
                    plt.tight_layout()
                    st.session_state.fig = fig
                    st.success("Model tested successfully!")
                except Exception as e:
                    st.error(f"Testing failed: {str(e)}")

# Results Section
if st.session_state.metrics or st.session_state.fig or st.session_state.train_results_df or st.session_state.test_results_df:
    with st.expander("📊 Results", expanded=True):
        if st.session_state.metrics is not None:
            st.subheader("📏 Performance Metrics")
            metrics_df = pd.DataFrame({
                "Metric": st.session_state.selected_metrics,
                "Training": [f"{st.session_state.metrics[m]['Training']:.4f}" for m in st.session_state.selected_metrics],
                "Testing": [f"{st.session_state.metrics[m]['Testing']:.4f}" for m in st.session_state.selected_metrics]
            })
            st.table(metrics_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]}
            ]))
        else:
            st.info("No results yet. Train and test the model to see metrics and plots.")

        col_plot, col_dl = st.columns([3, 1])
        with col_plot:
            if st.session_state.fig:
                st.subheader("📈 Prediction Plots")
                st.pyplot(st.session_state.fig)
        with col_dl:
            if st.session_state.fig:
                buf = BytesIO()
                st.session_state.fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                st.download_button("⬇️ Download Plot", buf.getvalue(), "prediction_plot.png", "image/png", key="plot_dl")
            if st.session_state.train_results_df is not None:
                train_csv = st.session_state.train_results_df.to_csv(index=False)
                st.download_button("⬇️ Train Data CSV", train_csv, "train_predictions.csv", "text/csv", key="train_dl")
            if st.session_state.test_results_df is not None:
                test_csv = st.session_state.test_results_df.to_csv(index=False)
                st.download_button("⬇️ Test Data CSV", test_csv, "test_predictions.csv", "text/csv", key="test_dl")

# New Data Prediction Section
if os.path.exists(MODEL_WEIGHTS_PATH):
    with st.expander("🔮 New Predictions", expanded=False):
        st.subheader("Predict New Data")
        new_data_file = st.file_uploader("Upload New Data (Excel)", type=["xlsx"], key="new_data")
        
        if new_data_file and new_data_file != st.session_state.new_data_file:
            st.session_state.new_data_file = new_data_file
            st.session_state.new_predictions_df = None
            st.session_state.new_fig = None
            st.session_state.selected_inputs = []
            st.session_state.new_date_col = None

        if st.session_state.new_data_file:
            new_df = pd.read_excel(st.session_state.new_data_file)
            st.write("**New Data Preview:**", new_df.head())
            
            datetime_cols = [col for col in new_df.columns if pd.api.types.is_datetime64_any_dtype(new_df[col]) or "date" in col.lower()]
            if datetime_cols:
                if st.session_state.new_date_col is None:
                    st.session_state.new_date_col = datetime_cols[0]
                date_col = st.selectbox("Select Date Column", datetime_cols, index=datetime_cols.index(st.session_state.new_date_col) if st.session_state.new_date_col in datetime_cols else 0, key="date_col_new")
                st.session_state.new_date_col = date_col
                new_df[date_col] = pd.to_datetime(new_df[date_col])
                new_df = new_df.sort_values(date_col)
            else:
                st.warning("No datetime column found. Predictions will use index.")
                date_col = None
            
            input_vars = st.session_state.input_vars
            output_var = st.session_state.output_var
            training_cols = [output_var] + input_vars + st.session_state.feature_cols
            available_new_inputs = [col for col in new_df.columns if col in input_vars and (date_col is None or col != date_col)]
            if not available_new_inputs:
                st.error("No recognized input variables found. Include: " + ", ".join(input_vars))
            else:
                if st.session_state.selected_inputs == []:
                    st.session_state.selected_inputs = available_new_inputs
                selected_inputs = st.multiselect("🔧 Input Variables for Prediction", available_new_inputs, default=[col for col in available_new_inputs if col in st.session_state.selected_inputs] if available_new_inputs else [], key="new_input_vars")
                st.session_state.selected_inputs = selected_inputs
                
                if not selected_inputs:
                    st.error("Select at least one input variable.")
                elif st.button("🔍 Predict"):
                    feature_cols = []
                    for var in selected_inputs:
                        for lag in range(1, NUM_LAGGED_FEATURES + 1):
                            if f'{var}_Lag_{lag}' not in new_df.columns:
                                new_df[f'{var}_Lag_{lag}'] = new_df[var].shift(lag)
                            feature_cols.append(f'{var}_Lag_{lag}')
                    new_df.dropna(inplace=True)
                    
                    # Match training column order
                    full_new_df = pd.DataFrame(index=new_df.index, columns=training_cols)
                    full_new_df[output_var] = 0
                    for col in selected_inputs:
                        if col in new_df.columns:
                            full_new_df[col] = new_df[col]
                    for col in feature_cols:
                        if col in new_df.columns:
                            full_new_df[col] = new_df[col]
                    full_new_df = full_new_df.fillna(0)
                    
                    # Ensure all training columns are present, fill missing with zeros
                    for col in training_cols:
                        if col not in full_new_df.columns:
                            full_new_df[col] = 0
                    
                    scaler = st.session_state.scaler
                    new_scaled = scaler.transform(full_new_df[training_cols])
                    X_new = new_scaled[:, 1:1 + len(feature_cols)]
                    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
                    
                    model = build_gru_model((X_new.shape[1], X_new.shape[2]), st.session_state.gru_layers, st.session_state.dense_layers, st.session_state.gru_units, st.session_state.dense_units, st.session_state.learning_rate)
                    model.load_weights(MODEL_WEIGHTS_PATH)
                    y_new_pred = model.predict(X_new)
                    y_new_pred_full = np.zeros((len(y_new_pred), len(training_cols)))
                    y_new_pred_full[:, 0] = y_new_pred.flatten()
                    y_new_pred_full[:, 1:] = new_scaled[:, 1:]
                    y_new_pred_inv = scaler.inverse_transform(y_new_pred_full)[:, 0]
                    y_new_pred = np.clip(y_new_pred_inv, 0, None)
                    
                    dates = new_df[date_col] if date_col else pd.RangeIndex(len(new_df))
                    st.session_state.new_predictions_df = pd.DataFrame({
                        "Date": dates.values[-len(y_new_pred):],
                        f"Predicted_{output_var}": y_new_pred
                    })
                    
                    # Single subplot for new predictions
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(dates.values[-len(y_new_pred):], y_new_pred, label="Predicted", color="#ff7f0e", linewidth=2)
                    if date_col != "None":
                        ax.set_xlabel("Date")
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    else:
                        ax.set_xlabel("Index")
                    ax.set_title(f"New Predictions: {output_var}", fontsize=14, pad=10)
                    ax.set_ylabel(output_var)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.session_state.new_fig = fig
                
                if st.session_state.new_predictions_df is not None:
                    st.subheader("Prediction Results")
                    st.write(st.session_state.new_predictions_df)
                    col_new_plot, col_new_dl = st.columns([3, 1])
                    with col_new_plot:
                        if st.session_state.new_fig:
                            st.pyplot(st.session_state.new_fig)
                    with col_new_dl:
                        if st.session_state.new_fig:
                            buf = BytesIO()
                            st.session_state.new_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                            st.download_button("⬇️ Download Plot", buf.getvalue(), "new_prediction_plot.png", "image/png", key="new_plot_dl")
                        if st.session_state.new_predictions_df is not None:
                            new_csv = st.session_state.new_predictions_df.to_csv(index=False)
                            st.download_button("⬇️ Download CSV", new_csv, "new_predictions.csv", "text/csv", key="new_csv_dl")
                    st.success("Predictions generated successfully!")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ by xAI | Powered by GRU and Streamlit**")
