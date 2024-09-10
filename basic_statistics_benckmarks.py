import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from basic_statistics_forks import calculate_fork_statistics

def extract_names(file_name):
    file_name = file_name.replace('.json', '')

    # Extract system_name
    separator_index = file_name.find('#')
    system_name = file_name[:separator_index]

    # Extract benchmark_name as the substring between the last dot and the '.json'
    benchmark_name = file_name[file_name.rfind('.') + 1:]

    return system_name, benchmark_name

def load_benchmark_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Calculate benchmark-level statistics by averaging fork-level statistics
def calculate_benchmark_statistics(benchmark_data):
    fork_stats = calculate_fork_statistics(benchmark_data).drop(columns=['Fork'])
    benchmark_stats = {key: np.mean(values) for key, values in fork_stats.items()}
    return benchmark_stats

# Plot the statistics table for a given system with benchmarks and statistics
def plot_statistics_table(system_name, benchmarks_stats):
    stats_df = pd.DataFrame.from_dict(benchmarks_stats, orient='index')
    
    fig, ax = plt.subplots(figsize=(16, len(stats_df) * 0.6))
    ax.axis('tight')
    ax.axis('off')

    # Create the table with the benchmark names and statistics
    cell_text = []
    for index, row in stats_df.iterrows():
        cell_text.append([
            index, 
            *["{:.7f}".format(row[col]) for col in stats_df.columns]
        ])
    
    table = ax.table(
        cellText=cell_text,
        colLabels=['Benchmark', *stats_df.columns],
        cellLoc='center',
        loc='center'
    )
    
    # Adjust table scaling for better readability
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Specifically adjust the width of the first column (benchmark names)
    for (i, j), cell in table.get_celld().items():
        if j == 0:
            cell.set_width(0.3)

    fig.suptitle(f"Statistics for {system_name}", fontsize=14)

    # Save the table as a PNG image
    image_path = os.path.join(output_folder, f"{system_name}_statistics.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Table saved as: {image_path}")


# Process all benchmarks in the data folder and categorize by system
def process_all_benchmarks(data_folder):
    systems_data = {}

    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):
            system_name, benchmark_name = extract_names(file_name)
            file_path = os.path.join(data_folder, file_name)
            benchmark_data = load_benchmark_data(file_path)

            benchmark_stats = calculate_benchmark_statistics(benchmark_data)
            if system_name not in systems_data:
                systems_data[system_name] = {}
            systems_data[system_name][benchmark_name] = benchmark_stats

    for system_name, benchmarks_stats in systems_data.items():
        plot_statistics_table(system_name, benchmarks_stats)

# data_folder = './timeseries'
# output_folder = './output/statistics'
# process_all_benchmarks(data_folder)