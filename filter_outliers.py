import json
import os
import pandas as pd

from basic_statistics_benckmarks import load_benchmark_data

# Use the method by Tukey to filter outliers
def filter_outliers(data, window_size=200):
    data = pd.Series(data)
    is_valid = pd.Series([True] * len(data))  # Create a boolean mask for valid points

    # Iterate through the data with a rolling window
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        median = window.median()
        lower_bound = median - 3 * (window.quantile(0.99) - window.quantile(0.01))
        upper_bound = median + 3 * (window.quantile(0.99) - window.quantile(0.01))

        # Mark outliers as False in the is_valid mask
        is_valid[i:i + window_size] = is_valid[i:i + window_size] & (window >= lower_bound) & (window <= upper_bound)

    # Filter out the outliers using the boolean mask
    filtered_data = data[is_valid].to_numpy()

    # outlier_percentage = (1 - len(filtered_data) / len(data)) * 100
    print(f"Num of Outliers removed: {len(data) - len(filtered_data)}.")
    return filtered_data

def load_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_filtered_data(file_path, data):
    data_as_list = [fork.tolist() for fork in data]
    with open(file_path, 'w') as file:
        json.dump(data_as_list, file, indent=4)

def process_benchmarks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_path = os.path.join(input_folder, file_name)
            benchmark_data = load_benchmark_data(input_path)

            filtered_benchmark_data = [filter_outliers(fork) for fork in benchmark_data]

            output_path = os.path.join(output_folder, file_name)
            save_filtered_data(output_path, filtered_benchmark_data)

            print(f"Processed {file_name} and saved filtered data to {output_path}.")

input_folder = './timeseries'
output_folder = './output/filtered_data'

process_benchmarks(input_folder, output_folder)