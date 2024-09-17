import os
import json
import numpy as np

# Use the approach of Kalibera and Jones to detect the steady state
def find_steady_state_index(data, change_points, steady_segment_size=500, tolerance=0.05):
    last_segment = data[change_points[-2]:change_points[-1]]
    steady_state_index = len(data)

    # Compare back segments with the last segment
    for i in reversed(range(len(change_points) - 2)):
        current_segment = data[change_points[i]:change_points[i + 1]]

        # Check if mean relative performance change is within tolerance
        mean_diff = abs(np.mean(last_segment) - np.mean(current_segment))
        ci_diff = np.mean(last_segment) * tolerance

        if mean_diff <= ci_diff:
            steady_state_index = change_points[i]
            continue
        else:
            steady_state_index = change_points[i + 1]

    # Check if steady segments are bigger than 500
    reached_steady = steady_state_index < (len(data) - steady_segment_size)

    print(f"Steady state begins at index: {steady_state_index}")
    print(f"Steady state reached: {'Yes' if reached_steady else 'No'}")

    return steady_state_index, reached_steady

def classify_benchmark(filtered_data, change_points):
    fork_classifications = []
    steady_state_starts = []

    for data, cps in zip(filtered_data, change_points):
        steady_index, is_steady = find_steady_state_index(data, cps)
        steady_state_starts.append(steady_index)
        fork_classifications.append("steady state" if is_steady else "no steady state")

    # Classify the overall benchmark
    if all(fork == "steady state" for fork in fork_classifications):
        overall_class = "steady state"
    elif any(fork == "steady state" for fork in fork_classifications):
        overall_class = "inconsistent"
    else:
        overall_class = "no steady state"

    return {
        "class": overall_class,
        "forks": fork_classifications,
        "steady_state_starts": steady_state_starts
    }

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_classification(file_path, classification):
    with open(file_path, 'w') as file:
        json.dump(classification, file, indent=4)

def process_benchmarks(filtered_data_folder, change_points_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all JSON files in the filtered data directory
    for file_name in os.listdir(change_points_folder):
        if file_name.endswith('.json'):
            # Construct file paths
            filtered_data_path = os.path.join(filtered_data_folder, file_name)
            change_points_path = os.path.join(change_points_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Load filtered data and change points
            filtered_data = load_json(filtered_data_path)
            change_points = load_json(change_points_path)

            classification = classify_benchmark(filtered_data, change_points)
            save_classification(output_path, classification)

            print(f"Processed {file_name} and saved classification to {output_path}.")

filtered_data_folder = './output/filtered_data'
change_points_folder = './output/change_points'
output_folder = './output/classification'

process_benchmarks(filtered_data_folder, change_points_folder, output_folder)