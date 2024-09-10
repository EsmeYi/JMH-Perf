from math import log
import json
import os
import numpy as np
import ruptures as rpt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler

# CROPS + Kneedle for optimal penalty detection
def find_penalty(data, penalties):
    default_pen = 15 * log(len(data))
    # Use the CROPS algorithm to detect change points with varying penalties
    algo = rpt.Pelt(model="l2").fit(data)
    all_change_points = [algo.predict(pen=pen) for pen in penalties]

    # Calculate the cost for each penalty (used for Kneedle algorithm)
    costs = [algo.cost.sum_of_costs(cp) for cp in all_change_points]

    # Use the Kneedle algorithm to find the elbow point
    try:
        kneedle = KneeLocator(penalties, costs, curve='convex', direction='decreasing')
        optimal_penalty = kneedle.elbow if kneedle.elbow else default_pen
    except IndexError:
        optimal_penalty = default_pen

    return optimal_penalty

def standardize_data(data):
    scaler = StandardScaler()
    data_reshaped = data.reshape(-1, 1)
    standardized_data = scaler.fit_transform(data_reshaped)
    return standardized_data.flatten()

# Use PELT algorithm to detect change points
def detect_change_points(data):
    # Preprocess data
    data = np.array(data)
    data = standardize_data(data)

    # Generate a range of penalties
    penalty_range=(4, 100000)
    penalties = np.logspace(np.log10(penalty_range[0]), np.log10(penalty_range[1]), num=100)
    optimal_penalty = find_penalty(data, penalties)

    algo = rpt.Pelt(model='l2').fit(data)
    change_points = algo.predict(pen=optimal_penalty)

    return change_points

def load_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_change_points(file_path, change_points):
    with open(file_path, 'w') as file:
        json.dump(change_points, file, indent=4)

def process_benchmarks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_path = os.path.join(input_folder, file_name)
            benchmark_data = load_benchmark_data(input_path)

            # Detect change points for each fork
            all_change_points = [detect_change_points(fork) for fork in benchmark_data]

            # Save the detected change points to a new file in the output directory
            output_path = os.path.join(output_folder, file_name)
            save_change_points(output_path, all_change_points)

            print(f"Processed {file_name} and saved change points to {output_path}.")


input_folder = './output/filtered_data/apache'
output_folder = './output/change_points'

process_benchmarks(input_folder, output_folder)