


import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import json

# Load the CSV file, skipping bad lines and handling quoting
df = pd.read_csv(
    'results/study-results-combined.csv',
    on_bad_lines='skip',
    quoting=csv.QUOTE_MINIMAL
)

# Define clips and questions
clips = ['clip1', 'clip2', 'clip3']
questions = ['clarity', 'understandable', 'completeness', 'satisfaction', 'useful', 'accuracy', 'improvement']
methods = ['video', 'image', 'gradcam']

# Prepare data structures for aggregation
clip_results = {clip: {method: {q: 0 for q in questions} for method in methods} for clip in clips}
clip_counts = {clip: {method: {q: 0 for q in questions} for method in methods} for clip in clips}

comprehensive_methods = ['video', 'image', 'highlights']
comprehensive_results = {method: [] for method in comprehensive_methods}

# Parse each row
for idx, row in df.iterrows():
    combined = row.get('combined-results')
    if not isinstance(combined, str):
        continue
    try:
        combined_json = json.loads(combined)
    except Exception:
        continue

    # Try both possible keys for short study
    short_study = combined_json.get('shortStudy') or combined_json.get('short_study')
    if short_study:
        # Try both possible keys for clips
        clips_data = short_study.get('clips')
        if clips_data:
            # Newer format: list of clips with rankings
            for i, clip_data in enumerate(clips_data):
                clip_name = f'clip{i+1}'
                rankings = clip_data.get('rankings', {})
                for q in questions:
                    q_rank = rankings.get(q, {})
                    for rank, info in q_rank.items():
                        method = info.get('methodType')
                        if method in methods:
                            clip_results[clip_name][method][q] += int(rank)
                            clip_counts[clip_name][method][q] += 1
        else:
            # Older format: keys like 'clip1_rankings'
            for clip_name in clips:
                clip_rankings = short_study.get(f'{clip_name}_rankings', {})
                for q in questions:
                    q_rank = clip_rankings.get(q, [])
                    for rank_idx, method in enumerate(q_rank):
                        if method in methods:
                            clip_results[clip_name][method][q] += rank_idx + 1
                            clip_counts[clip_name][method][q] += 1

    # Comprehensive study
    comp = combined_json.get('comprehensiveStudy') or combined_json.get('comprehensive_study')
    if comp:
        ratings = comp.get('methodRatings') or comp.get('ratings')
        if ratings:
            for method in comprehensive_methods:
                val = ratings.get(method)
                if val is not None:
                    comprehensive_results[method].append(val)

# Aggregate and plot for each clip

for clip_name in clips:
    # Prepare mean and std data
    mean_data = {method: [] for method in methods}
    std_data = {method: [] for method in methods}
    # For variance, collect all rankings per method/question
    rankings_per_method_q = {method: {q: [] for q in questions} for method in methods}

    for idx, row in df.iterrows():
        combined = row.get('combined-results')
        if not isinstance(combined, str):
            continue
        try:
            combined_json = json.loads(combined)
        except Exception:
            continue
        short_study = combined_json.get('shortStudy') or combined_json.get('short_study')
        if short_study:
            clips_data = short_study.get('clips')
            if clips_data:
                # Newer format
                i = int(clip_name[-1]) - 1
                if i < len(clips_data):
                    rankings = clips_data[i].get('rankings', {})
                    for q in questions:
                        q_rank = rankings.get(q, {})
                        for rank, info in q_rank.items():
                            method = info.get('methodType')
                            if method in methods:
                                rankings_per_method_q[method][q].append(int(rank))
            else:
                # Older format
                clip_rankings = short_study.get(f'{clip_name}_rankings', {})
                for q in questions:
                    q_rank = clip_rankings.get(q, [])
                    for rank_idx, method in enumerate(q_rank):
                        if method in methods:
                            rankings_per_method_q[method][q].append(rank_idx + 1)

    for method in methods:
        for q in questions:
            vals = rankings_per_method_q[method][q]
            mean = sum(vals) / len(vals) if vals else 0
            std = pd.Series(vals).std() if vals else 0
            mean_data[method].append(mean)
            std_data[method].append(std)

    df_plot = pd.DataFrame(mean_data, index=questions)
    df_std = pd.DataFrame(std_data, index=questions)
    ax = df_plot.plot(kind='bar', yerr=df_std, capsize=4, figsize=(12, 8), legend=True)
    plt.title(f'{clip_name.capitalize()} - Rankings for 7 Questions')
    plt.ylabel('Average Ranking (lower is better)')
    plt.xlabel('Question')
    plt.xticks(rotation=30)
    # Move legend to upper right outside plot
    plt.legend(title='Method', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.figtext(0.99, 0.01, 'Lower is better. Error bars show standard deviation.', ha='right', fontsize=10, color='gray')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'results/{clip_name}_scores.png', bbox_inches='tight')
    plt.close()


# Plot for comprehensive study (show mean and std)
comp_means = {method: (sum(vals)/len(vals) if vals else 0) for method, vals in comprehensive_results.items()}
comp_stds = {method: (pd.Series(vals).std() if vals else 0) for method, vals in comprehensive_results.items()}
df_comp = pd.DataFrame([comp_means], index=['Comprehensive'])
df_comp_std = pd.DataFrame([comp_stds], index=['Comprehensive'])
ax = df_comp.plot(kind='bar', yerr=df_comp_std, capsize=4, figsize=(10, 7), legend=True)
plt.title('Comprehensive Study - Method Ratings')
plt.ylabel('Average Rating')
plt.xlabel('Method')
plt.xticks(rotation=0)
# Move legend to upper right outside plot
plt.legend(title='Method', loc='upper right', bbox_to_anchor=(1.15, 1))
plt.figtext(0.99, 0.01, 'Higher is better. Error bars show standard deviation.', ha='right', fontsize=10, color='gray')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('results/comprehensive_scores.png', bbox_inches='tight')
plt.close()

print('Plots saved to results/*.png')
