import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from basic_statistics_benckmarks import extract_names

classification_dir = "./configs/classification/"
devconfig_dir = "./configs/devconfig/"
filtered_data_dir = "./output/filtered_data/"

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# def calculate_rpd(stable_measurements, all_measurements, confidence_level=0.95):
#     M_stable = np.mean(stable_measurements)
#     M_all = np.mean(all_measurements)

#     # Calculate mean relative performance change
#     relative_change = (M_stable - M_all) / M_all * 100

#     # Calculate standard error of the relative performance change
#     combined_std = np.sqrt(
#         (np.std(stable_measurements, ddof=1) ** 2 / len(stable_measurements)) +
#         (np.std(all_measurements, ddof=1) ** 2 / len(all_measurements))
#     )
#     relative_std_error = combined_std / M_all * 100

#     # Calculate the 95% confidence interval
#     dof = len(stable_measurements) + len(all_measurements) - 2
#     t_value = t.ppf((1 + confidence_level) / 2, dof)
#     ci_lower = relative_change - t_value * relative_std_error
#     ci_upper = relative_change + t_value * relative_std_error

#     # Determine the RPD based on the confidence interval
#     if ci_lower > 0 or ci_upper < 0:
#         rpd = (ci_upper + ci_lower) / 2
#     else:  # Confidence interval contains zero
#         rpd = 0

#     return rpd

def calculate_rpd(m_stable, m_all):
    return abs(np.mean(m_stable) - np.mean(m_all)) / np.mean(m_all) * 100

results = []

for file_name in os.listdir(classification_dir):
    if file_name.endswith(".json"):
        classification = load_json(os.path.join(classification_dir, file_name))
        devconfig = load_json(os.path.join(devconfig_dir, file_name))
        filtered_data = load_json(os.path.join(filtered_data_dir, file_name))

        # Only process inconsistent benchmarks
        if classification["run"] == "inconsistent":
            system_name, benchmark_name = extract_names(file_name)

            # Identify steady forks
            steady_forks_indices = [
                i for i, fork_status in enumerate(classification["forks"]) if fork_status == "steady state"
            ]

            # Gather measurements for M(stable-only)
            m_stable = []
            for i in steady_forks_indices:
                for seg in devconfig:
                    start, end = seg
                    m_stable.extend(filtered_data[i][start:end + 1])
            m_stable = np.array(m_stable)

            # Gather measurements for M(all-set)
            m_all = []
            for i in range(len(filtered_data)):
                for seg in devconfig:
                    start, end = seg
                    m_all.extend(filtered_data[i][start:end + 1])
            m_all = np.array(m_all)

            # Calculate RPD
            if len(m_stable) > 0 and len(m_all) > 0:
                rpd_value = calculate_rpd(m_stable, m_all)
                results.append({
                    "System Name": system_name,
                    "Benchmark Name": benchmark_name,
                    "RPD (%)": rpd_value
                })

df_results = pd.DataFrame(results)

grouped_results = df_results.groupby("System Name")

## Save tables for each system
# for system_name, group in grouped_results:
#     fig, ax = plt.subplots(figsize=(10, min(5, len(group) * 0.5)))
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(
#         cellText=group.values,
#         colLabels=group.columns,
#         cellLoc='center',
#         loc='center'
#     )
#     table.auto_set_column_width(col=[0, 1, 2])
#     ax.set_title(f'RPD for Inconsistent Benchmarks - {system_name}', fontsize=14, pad=20)
#     plt.savefig(f"./output/statistics_inconsistent_benchmarks/RPD_Table_{system_name}.png", bbox_inches="tight")
#     plt.show()

# Plot boxplots for grouped RPD results
plt.figure(figsize=(12, 8))

boxplot_data = [group["RPD (%)"].values for _, group in grouped_results]
system_names = [name for name, _ in grouped_results]

plt.boxplot(boxplot_data, labels=system_names, patch_artist=True, notch=True, showmeans=True)

plt.xlabel("Projects")
plt.ylabel("Performance Changes (%)")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("RPD_Boxplot_Grouped_by_Project.png")
plt.show()