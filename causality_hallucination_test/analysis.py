import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUTPUT_DIR = Path("causality_hallucination_test")

def analyze_results():
    import json
    results_path = OUTPUT_DIR / "inference_results" / "all_inference_results.json"
    with open(results_path, "r") as f:
        all_results = json.load(f)

    conditions = ["condition_A", "condition_B", "condition_C"]
    all_predictions = {c: [] for c in conditions}
    all_ground_truths = {c: [] for c in conditions}

    for item in all_results:
        cond = item.get("condition")
        try:
            pred = int(item.get("prediction"))
            gt = int(item.get("groundtruth")) if item.get("groundtruth") is not None else None
        except (ValueError, TypeError):
            pred = None
            gt = None
        if cond in conditions:
            if pred is not None:
                all_predictions[cond].append(pred)
            if gt is not None:
                all_ground_truths[cond].append(gt)

    # Plot three separate figures for each condition
    for cond in conditions:
        plt.figure(figsize=(8, 5))
        plt.hist(all_predictions[cond], bins=20, alpha=0.6, label=f"Predictions", color='blue')
        plt.hist(all_ground_truths[cond], bins=20, alpha=0.4, label=f"Ground Truths", color='orange')
        plt.xlabel("Landing Position (X)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {cond}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"results_{cond}.png")
        plt.close()


if __name__ == "__main__":
    analyze_results()