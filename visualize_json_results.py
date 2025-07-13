import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to the results folder
RESULTS_DIR = 'json_results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# List all JSON files in the results directory
json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]

# Collect all evaluation data
all_evaluations = []

# Individual file visualizations
for json_file in json_files:
    file_path = os.path.join(RESULTS_DIR, json_file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract methods and their evaluation metrics
    methods = [entry['method'] for entry in data]
    metrics = list(data[0]['evaluations'].keys())
    
    # Prepare data for plotting
    values = []
    for entry in data:
        values.append([entry['evaluations'][metric] for metric in metrics])
    values = np.array(values)
    
    # Plotting
    x = np.arange(len(metrics))  # the label locations
    width = 0.8 / len(methods)   # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        ax.bar(x + i * width, values[i], width, label=method)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Results: {json_file}')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=30)
    ax.legend()
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plot_filename = os.path.splitext(json_file)[0] + '.png'
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    
    # Collect evaluations for summary
    all_evaluations.extend(data)

# Aggregate Results
# Create a DataFrame for easier aggregation
df = pd.DataFrame([
    {**entry, **entry['evaluations']} 
    for entry in all_evaluations
])

# Group by method and compute mean of each metric
summary_df = df.groupby('method')[list(df['evaluations'].iloc[0].keys())].mean()

# Visualization of Aggregated Results with Grouped Bar Chart
plt.figure(figsize=(12, 6))

# Prepare data for grouped bar chart
metrics = summary_df.columns
methods = summary_df.index
x = np.arange(len(metrics))  # the label locations
width = 0.8 / len(methods)   # the width of the bars

# Create the grouped bar chart
for i, method in enumerate(methods):
    plt.bar(x + i * width, summary_df.loc[method], width, label=method)

plt.title('Mean Evaluation Scores Across All Files')
plt.xlabel('Metrics')
plt.ylabel('Mean Score')
plt.xticks(x + width * (len(methods) - 1) / 2, metrics, rotation=30)
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'aggregated_summary_grouped.png'))
plt.close()

# Print summary for text-based overview
print("Aggregated Mean Scores:")
print(summary_df) 