import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_benchmark_data():
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Visualize forks of a benchmark
def vis_forks_benchmark(benchmark_data):        
    plt.figure(figsize=(15, 12))
    for i, fork_data in enumerate(benchmark_data):
        plt.subplot(5, 2, i + 1)
        plt.plot(fork_data, marker='.', linestyle='-', color='b')
        plt.title(f'Fork {i + 1}')
        plt.xlabel('Iterations')
        plt.ylabel('Execution time (sec)')
        plt.grid(True)

    plt.tight_layout()
    # Save the figure as a PNG image
    image_path = os.path.join(output_folder, f"vis_time_series.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

# Visualize boxplot for each fork
def plot_forks_boxplot(benchmark_data):
    plt.figure(figsize=(12, 6))
    plt.boxplot(benchmark_data, notch=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.xlabel('Fork Number')
    plt.ylabel('Execution time (sec)')
    plt.grid(True)

    # Save the figure as a PNG image
    image_path = os.path.join(output_folder, f"forks_boxplot.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

# Steady State Error for each fork
def calculate_steady_state_error(fork):
    time_series = np.array(fork)
    steady_state_values = time_series[-300:]
    steady_state_mean = np.mean(steady_state_values)
    steady_state_error = np.mean(np.abs(time_series - steady_state_mean))
    return steady_state_error

# Coefficient of Variation for each fork
def calculate_cv(fork):
    mean = np.mean(fork)
    std_dev = np.std(fork)
    cv = std_dev / mean if mean != 0 else 0
    return cv

# Statistics on fork level
def calculate_fork_statistics(benchmark_data):
    fork_stats = {
        'Fork': [],
        'Mean': [],
        'Std Dev': [],
        'Median': [],
        'Min': [],
        'Max': [],
        'Q1': [],
        'Q3': [],
        'SSE': [], # Steady State Error
        'CV': [] # Coefficient of Variation
    }

    for i, fork in enumerate(benchmark_data):
        steady_state_error = calculate_steady_state_error(fork)
        cv = calculate_cv(fork)
        fork_data = np.array(fork)
        fork_stats['Fork'].append(f"Fork {i+1}")
        fork_stats['Mean'].append(np.mean(fork_data))
        fork_stats['Std Dev'].append(np.std(fork_data))
        fork_stats['Median'].append(np.median(fork_data))
        fork_stats['Min'].append(np.min(fork_data))
        fork_stats['Max'].append(np.max(fork_data))
        fork_stats['Q1'].append(np.percentile(fork_data, 25))
        fork_stats['Q3'].append(np.percentile(fork_data, 75))
        fork_stats['SSE'].append(steady_state_error)
        fork_stats['CV'].append(cv)

    return pd.DataFrame(fork_stats)

# Visualize statistics table
def plot_statistics_table_fork(benchmark_data, file_name):
    stats_df = calculate_fork_statistics(benchmark_data)

    # Format to 7 decimal
    formatted_values = [
        [row[0]] + [f"{val:.7f}" for val in row[1:]]
        for row in np.column_stack((stats_df['Fork'], stats_df.drop(columns=['Fork']).values))
    ]

    fig, ax = plt.subplots(figsize=(14, len(stats_df) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=formatted_values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    fig.suptitle(f"Statistics for {file_name}", fontsize=14)
    # Save the figure as a PNG image
    image_path = os.path.join(output_folder, f"forks_statistics_table.png")
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_forks():
    benchmark_data = load_benchmark_data()
    file_name = file_path.split('/')[-1]
    print('Analyzing ', file_name, '..')
    vis_forks_benchmark(benchmark_data)
    plot_forks_boxplot(benchmark_data)
    plot_statistics_table_fork(benchmark_data, file_name)


file_path = './timeseries/apache__arrow#org.apache.arrow.adapter.jdbc.JdbcAdapterBenchmarks.consumeBenchmark#.json'
output_folder = './output/statistics_fork_example'
visualize_forks()