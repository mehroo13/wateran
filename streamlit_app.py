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
DEFAULT_TRAIN_SPLIT = 80
NUM_LAGGED_FEATURES = 3
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
st.set_page_config(page_title="Wateran", page_icon="📈", layout="wide")
st.title("🌊 Wateran")
st.markdown("**Design, train, and predict time series data effortlessly with GRU!**")

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
    st.session_state.input_vars = None
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
    st.session_state.feature_cols = None
if 'new_data_file' not in st.session_state:
    st.session_state.new_data_file = None
if 'selected_inputs' not in st.session_state:
    st.session_state.selected_inputs = None
if 'new_date_col' not in st.session_state:
    st.session_state.new_date_col = None

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    with st.expander("📥 Upload Training Data", expanded=True):
        uploaded_file = st.file_uploader("Choose an Excel file for training", type=["xlsx"], key="train_data")

with col2:
    with st.expander("⚙️ Model Settings", expanded=True):
        epochs = st.slider("Epochs:", 1, 1500, DEFAULT_EPOCHS, step=10)
        batch_size = st.slider("Batch Size:", 8, 128, DEFAULT_BATCH_SIZE, step=8)
        train_split = st.slider("Training Data %:", 50, 90, DEFAULT_TRAIN_SPLIT) / 100
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

# Process training data if uploaded
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    with col1:
        st.write("**Dataset Preview:**", df.head())
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower()]
        if datetime_cols:
            date_col = st.selectbox("Select datetime column (optional):", ["None"] + datetime_cols, index=0, key="date_col_train")
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
            else:
                st.info("Using index for ordering (assuming sequential data).")
        else:
            st.info("No datetime column found. Using index for ordering.")
            date_col = None
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and (date_col is None or col != date_col)]
        if len(numeric_cols) < 2:
            st.error("Dataset needs at least two numeric columns.")
            st.stop()
        output_var = st.selectbox("🎯 Output Variable to Predict:", numeric_cols, key="output_var_train")
        available_input_cols = [col for col in numeric_cols if col != output_var]
        if not available_input_cols:
            st.error("No input variables available. Please check your dataset.")
            st.stop()
        input_vars = st.multiselect("🔧 Input Variables:", available_input_cols, default=[available_input_cols[0]], key="input_vars_train")
        if not input_vars:
            st.error("Select at least one input variable.")
            st.stop()
        feature_cols = []
        for var in input_vars + [output_var]:
            for lag in range(1, NUM_LAGGED_FEATURES + 1):
                df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
                feature_cols.append(f'{var}_Lag_{lag}')
        df.dropna(inplace=True)
        st.session_state.input_vars = input_vars
        st.session_state.output_var = output_var
        st.session_state.gru_layers = gru_layers
        st.session_state.dense_layers = dense_layers
        st.session_state.gru_units = gru_units
        st.session_state.dense_units = dense_units
        st.session_state.learning_rate = learning_rate
        st.session_state.feature_cols = feature_cols
        st.session_state.date_col = date_col

    with col2:
        st.write(f"**Training Size:** {int(len(df) * train_split)} rows | **Testing Size:** {len(df) - int(len(df) * train_split)} rows")
        dummy_input_shape = (1, len(input_vars) + len(feature_cols))
        model = build_gru_model(dummy_input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate)
        try:
            plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True, dpi=96)
            st.image(MODEL_PLOT_PATH, caption="GRU Model Structure", use_container_width=True)
        except ImportError:
            st.warning("Install 'pydot' and 'graphviz' for visualization.")

    train_size = int(len(df) * train_split)
    train_df, test_df = df[:train_size], df[train_size:]
    all_feature_cols = input_vars + feature_cols
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[output_var] + all_feature_cols])
    test_scaled = scaler.transform(test_df[[output_var] + all_feature_cols])
    st.session_state.scaler = scaler
    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0]
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Train Model"):
            model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
            try:
                with st.spinner("Training in progress..."):
                    progress_placeholder = st.empty()
                    callback = StreamlitProgressCallback(epochs, progress_placeholder)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[callback])
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
            model = build_gru_model((X_train.shape[1], X_train.shape[2]), gru_layers, dense_layers, gru_units, dense_units, learning_rate)
            try:
                model.load_weights(MODEL_WEIGHTS_PATH)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_train_pred = scaler.inverse_transform(np.hstack([y_train_pred, X_train[:, 0, :]]))[:, 0]
                y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
                y_train_actual = scaler.inverse_transform(np.hstack([y_train.reshape(-1, 1), X_train[:, 0, :]]))[:, 0]
                y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]
                y_train_pred = np.clip(y_train_pred, 0, None)
                y_test_pred = np.clip(y_test_pred, 0, None)
                metrics = {
                    "Training RMSE": np.sqrt(mean_squared_error(y_train_actual, y_train_pred)),
                    "Testing RMSE": np.sqrt(mean_squared_error(y_test_actual, y_test_pred)),
                    "Training R²": r2_score(y_train_actual, y_train_pred),
                    "Testing R²": r2_score(y_test_actual, y_test_pred),
                    "Training NSE": nse(y_train_actual, y_train_pred),
                    "Testing NSE": nse(y_test_actual, y_test_pred)
                }
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
    with st.expander("📊 Training and Testing Results", expanded=True):
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
                st.download_button("⬇️ Training Data CSV", train_csv, "train_predictions.csv", "text/csv", key="train_dl")
            if st.session_state.test_results_df is not None:
                test_csv = st.session_state.test_results_df.to_csv(index=False)
                st.download_button("⬇️ Testing Data CSV", test_csv, "test_predictions.csv", "text/csv", key="test_dl")

# New Data Prediction Section
if os.path.exists(MODEL_WEIGHTS_PATH):
    with st.expander("🔮 Predict New Data", expanded=False):
        st.subheader("Upload New Data for Prediction")
        new_data_file = st.file_uploader("Choose an Excel file with new input data", type=["xlsx"], key="new_data")
        
        # Update session state with new data file only if it changes
        if new_data_file and new_data_file != st.session_state.new_data_file:
            st.session_state.new_data_file = new_data_file
            st.session_state.new_predictions_df = None
            st.session_state.new_fig = None
            st.session_state.selected_inputs = None
            st.session_state.new_date_col = None

        if st.session_state.new_data_file:
            new_df = pd.read_excel(st.session_state.new_data_file)
            st.write("**New Data Preview:**", new_df.head())
            
            # Date column handling for new data
            datetime_cols = [col for col in new_df.columns if pd.api.types.is_datetime64_any_dtype(new_df[col]) or "date" in col.lower()]
            if datetime_cols:
                if st.session_state.new_date_col is None:
                    st.session_state.new_date_col = datetime_cols[0]  # Default to first datetime column
                date_col = st.selectbox(
                    "Select datetime column for new data:",
                    datetime_cols,
                    index=datetime_cols.index(st.session_state.new_date_col) if st.session_state.new_date_col in datetime_cols else 0,
                    key="date_col_new"
                )
                st.session_state.new_date_col = date_col
                new_df[date_col] = pd.to_datetime(new_df[date_col])
                new_df = new_df.sort_values(date_col)
            else:
                st.warning("No datetime column found in new data. Predictions will not include dates.")
                date_col = None
            
            # Get input variables from training
            input_vars = st.session_state.input_vars
            output_var = st.session_state.output_var
            
            # Identify available input columns in new data
            available_new_inputs = [col for col in new_df.columns if col in input_vars and (date_col is None or col != date_col)]
            if not available_new_inputs:
                st.error("No recognized input variables found in the new data. Please include at least one of: " + ", ".join(input_vars))
            else:
                # Use cached selected inputs if available, otherwise set default
                if st.session_state.selected_inputs is None:
                    st.session_state.selected_inputs = available_new_inputs
                
                selected_inputs = st.multiselect(
                    "🔧 Select Input Variables for Prediction:",
                    available_new_inputs,
                    default=st.session_state.selected_inputs,
                    key="new_input_vars"
                )
                st.session_state.selected_inputs = selected_inputs
                
                if not selected_inputs:
                    st.error("Please select at least one input variable for prediction.")
                elif st.button("🔍 Predict"):
                    # Generate lagged features only for selected inputs
                    feature_cols = []
                    for var in selected_inputs:
                        for lag in range(1, NUM_LAGGED_FEATURES + 1):
                            new_df[f'{var}_Lag_{lag}'] = new_df[var].shift(lag)
                            feature_cols.append(f'{var}_Lag_{lag}')
                    new_df.dropna(inplace=True)
                    
                    # Prepare all features used in training
                    all_feature_cols = input_vars + st.session_state.feature_cols
                    new_all_feature_cols = selected_inputs + feature_cols
                    
                    # Create a DataFrame with all expected columns, filling missing ones with zeros
                    full_new_df = pd.DataFrame(index=new_df.index, columns=[output_var] + all_feature_cols)
                    full_new_df[output_var] = 0  # Dummy value for output, not used in prediction
                    for col in new_all_feature_cols:
                        full_new_df[col] = new_df[col]
                    full_new_df.fillna(0, inplace=True)  # Fill missing inputs/lags with 0
                    
                    # Scale the data
                    scaler = st.session_state.scaler
                    new_scaled = scaler.transform(full_new_df[[output_var] + all_feature_cols])
                    X_new = new_scaled[:, 1:]  # Exclude the dummy output column
                    X_new = X_new.reshape((X_new.shape[0], 1, X_new.shape[1]))
                    
                    # Predict
                    model = build_gru_model(
                        (X_new.shape[1], X_new.shape[2]),
                        st.session_state.gru_layers,
                        st.session_state.dense_layers,
                        st.session_state.gru_units,
                        st.session_state.dense_units,
                        st.session_state.learning_rate
                    )
                    model.load_weights(MODEL_WEIGHTS_PATH)
                    y_new_pred = model.predict(X_new)
                    y_new_pred = scaler.inverse_transform(np.hstack([y_new_pred, X_new[:, 0, :]]))[:, 0]
                    y_new_pred = np.clip(y_new_pred, 0, None)
                    
                    # Store predictions with dates if available
                    dates = new_df[date_col] if date_col else pd.RangeIndex(len(new_df))
                    st.session_state.new_predictions_df = pd.DataFrame({
                        "Date": dates.values[-len(y_new_pred):],  # Match length after dropping NaNs
                        f"Predicted_{output_var}": y_new_pred
                    })
                    
                    # Plot with dates
                    fig, ax = plt.subplots(figsize=(12, 4))
                    if date_col:
                        ax.plot(dates.values[-len(y_new_pred):], y_new_pred, label="Predicted", color="#ff7f0e", linewidth=2)
                        ax.set_xlabel("Date")
                        plt.xticks(rotation=45)
                    else:
                        ax.plot(y_new_pred, label="Predicted", color="#ff7f0e", linewidth=2)
                        ax.set_xlabel("Index")
                    ax.set_title(f"Predictions for New Data: {output_var}", fontsize=14, pad=10)
                    ax.set_ylabel(output_var)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.session_state.new_fig = fig
                
                # Display cached results if they exist
                if st.session_state.new_predictions_df is not None:
                    st.subheader("New Data Predictions")
                    st.write(st.session_state.new_predictions_df)
                    col_new_plot, col_new_dl = st.columns([3, 1])
                    with col_new_plot:
                        if st.session_state.new_fig:
                            st.pyplot(st.session_state.new_fig)
                    with col_new_dl:
                        if st.session_state.new_fig:
                            buf = BytesIO()
                            st.session_state.new_fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                            st.download_button("⬇️ Download New Prediction Plot", buf.getvalue(), "new_prediction_plot.png", "image/png", key="new_plot_dl")
                        if st.session_state.new_predictions_df is not None:
                            new_csv = st.session_state.new_predictions_df.to_csv(index=False)
                            st.download_button("⬇️ Download New Predictions CSV", new_csv, "new_predictions.csv", "text/csv", key="new_csv_dl")
                    st.success("Predictions generated successfully!")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ by xAI | Powered by GRU and Streamlit**")
