import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Function to load forks from a single benchmark file
def load_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data)  # Shape: (10, 3000)

# Function to create sequences for LSTM input
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Function to encode system types
def encode_system_types(system_names):
    le = LabelEncoder()
    system_encoded = le.fit_transform(system_names)
    system_encoded = np.eye(len(le.classes_))[system_encoded]  # One-hot encoding
    return system_encoded, le

# Prepare data from all benchmarks
def prepare_data(data_folder, n_steps):
    X, y, system_labels = [], [], []
    system_names = []

    # Iterate over each file to collect data and system types
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):
            # Extract system and benchmark name
            system_name = file_name.split('#')[0]
            file_path = os.path.join(data_folder, file_name)
            benchmark_data = load_benchmark_data(file_path)
            
            # Collect system names for encoding
            system_names.extend([system_name] * len(benchmark_data))  # One entry per fork
            
            # Normalize each fork independently and create sequences
            for fork in benchmark_data:
                scaler = MinMaxScaler(feature_range=(0, 1))
                fork_scaled = scaler.fit_transform(fork.reshape(-1, 1)).flatten()
                X_fork, y_fork = create_sequences(fork_scaled, n_steps)
                X.append(X_fork)
                y.append(y_fork)

    # Encode system types
    system_encoded, le = encode_system_types(system_names)

    # Reshape system labels to match each time series entry
    expanded_system_labels = []
    for i, system_encoding in enumerate(system_encoded):
        # Repeat the system encoding for each time step in the sequence
        num_samples = len(X[i])  # Get the number of sequences from the corresponding fork
        repeated_encoding = np.repeat(system_encoding.reshape(1, 1, -1), num_samples, axis=0)
        repeated_encoding = np.repeat(repeated_encoding, n_steps, axis=1)  # Match n_steps
        
        expanded_system_labels.append(repeated_encoding)

    # Convert lists to arrays and ensure shape compatibility
    X = np.concatenate(X, axis=0)  # Combine all forks
    y = np.concatenate(y, axis=0)
    expanded_system_labels = np.concatenate(expanded_system_labels, axis=0)
    
    # Ensure `X` has the correct shape for concatenation
    X = np.expand_dims(X, axis=2)  # Shape: (num_samples, n_steps, 1)
    
    # Combine system encoding with the time series data
    X_combined = np.concatenate([X, expanded_system_labels], axis=2)  # Shape: (num_samples, n_steps, num_features)
    return X_combined, y, le

# Split data into train and test sets
def split_data(X, y, train_size=0.8):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

# Define the LSTM model with system type as input
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, activation='relu')(inputs)
    outputs = Dense(1)(lstm_out)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# Main execution flow
data_folder = './timeseries/train_set'  # Path to your dataset folder
n_steps = 100  # Number of previous steps used for prediction

# Prepare the dataset
X, y, label_encoder = prepare_data(data_folder, n_steps)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Build and train the LSTM model
model = build_lstm_model((n_steps, X_train.shape[2]))
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=20, verbose=1, callbacks=[early_stopping])

# Predict on test data
y_pred = model.predict(X_test)

# Plotting a sample of actual vs. predicted
plt.figure(figsize=(14, 7))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.title('LSTM Prediction on Test Data with System Type')
plt.xlabel('Time Steps')
plt.ylabel('Performance Metric')
plt.legend()
plt.show()
