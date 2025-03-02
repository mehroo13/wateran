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

# -------------------- GRU Model with Custom Layers --------------------
def build_gru_model(input_shape, gru_layers, dense_layers, gru_units, dense_units, learning_rate, use_bi_gru):
    model = tf.keras.Sequential()

    # Add GRU or Bi-GRU layers
    for i in range(gru_layers):
        return_seq = (i < gru_layers - 1)
        if use_bi_gru:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units[i], return_sequences=return_seq), input_shape=input_shape))
        else:
            model.add(tf.keras.layers.GRU(gru_units[i], return_sequences=return_seq, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(0.2))  # Adding dropout for regularization

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
st.markdown("**Design and predict time series data with a customizable GRU model. Visualize your architecture in real-time!**")

# Sidebar for model parameters
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    epochs = st.slider("Epochs:", 1, 1500, DEFAULT_EPOCHS, step=10)
    batch_size = st.slider("Batch Size:", 8, 128, DEFAULT_BATCH_SIZE, step=8)
    train_split = st.slider("Training Data %:", 50, 90, DEFAULT_TRAIN_SPLIT) / 100
    learning_rate = st.number_input("Learning Rate:", min_value=0.00001, max_value=0.1, value=DEFAULT_LEARNING_RATE, format="%.5f")

    # Custom architecture
    st.subheader("Model Architecture")
    gru_layers = st.number_input("Number of GRU Layers:", min_value=1, max_value=5, value=2, step=1)
    gru_units = [st.number_input(f"GRU Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_GRU_UNITS, step=8, key=f"gru_{i}") for i in range(gru_layers)]
    
    dense_layers = st.number_input("Number of Dense Layers:", min_value=1, max_value=5, value=2, step=1)
    dense_units = [st.number_input(f"Dense Layer {i+1} Units:", min_value=8, max_value=512, value=DEFAULT_DENSE_UNITS, step=8, key=f"dense_{i}") for i in range(dense_layers)]

    use_bi_gru = st.checkbox("Use Bi-GRU Layer?", value=True)

# Upload data
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("**Dataset Preview:**", df.head())

    # Select numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    output_var = st.selectbox("🎯 Output Variable:", numeric_cols)
    input_vars = st.multiselect("🔧 Input Variables:", [col for col in numeric_cols if col != output_var], default=[numeric_cols[0]])

    if not input_vars:
        st.error("Select at least one input variable.")
        st.stop()

    # Lag features
    feature_cols = []
    for var in input_vars + [output_var]:
        for lag in range(1, NUM_LAGGED_FEATURES + 1):
            df[f'{var}_Lag_{lag}'] = df[var].shift(lag)
            feature_cols.append(f'{var}_Lag_{lag}')
    df.dropna(inplace=True)

    # Train-test split
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

    # Build model
    model = build_gru_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        gru_layers=gru_layers,
        dense_layers=dense_layers,
        gru_units=gru_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        use_bi_gru=use_bi_gru
    )

    # Save model plot
    try:
        plot_model(model, to_file=MODEL_PLOT_PATH, show_shapes=True, show_layer_names=True, dpi=96)
        st.image(MODEL_PLOT_PATH, caption="GRU Model Structure", use_column_width=True)
    except ImportError:
        st.warning("Install 'pydot' and 'graphviz' to enable model visualization.")

    # Train button
    if st.button("🚀 Train Model"):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        model.save_weights(MODEL_WEIGHTS_PATH)
        st.success("Model trained successfully!")

    # Test button
    if st.button("🔍 Test Model"):
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            st.error("Train the model first!")
        else:
            model.load_weights(MODEL_WEIGHTS_PATH)
            y_test_pred = model.predict(X_test)
            y_test_pred = scaler.inverse_transform(np.hstack([y_test_pred, X_test[:, 0, :]]))[:, 0]
            y_test_actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), X_test[:, 0, :]]))[:, 0]

            # Plot results
            fig, ax = plt.subplots()
            ax.plot(y_test_actual, label="Actual")
            ax.plot(y_test_pred, label="Predicted", linestyle="--")
            ax.legend()
            st.pyplot(fig)

st.markdown("---")
st.markdown("**Built with ❤️ by xAI | Powered by GRU and Streamlit**")
