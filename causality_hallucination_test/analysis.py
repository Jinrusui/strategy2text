import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUTPUT_DIR = Path("causality_hallucination_test")

def analyze_results():
    conditions = ["condition_A", "condition_B", "condition_C"]
    all_predictions = []
    all_ground_truths = []

    for condition in conditions:
        condition_dir = OUTPUT_DIR / condition
        predictions = []
        ground_truths = []
        for f in condition_dir.glob("*_ground_truth.txt"):
            with open(f, "r") as f_in:
                ground_truths.append(int(f_in.read()))
            
            prediction_file = f.with_name(f.stem.replace("_ground_truth", ".txt"))
            if prediction_file.exists():
                with open(prediction_file, "r") as f_in:
                    try:
                        predictions.append(int(f_in.read()))
                    except ValueError:
                        # Handle cases where the prediction is not a number (e.g., "Error")
                        pass

        all_predictions.append(predictions)
        all_ground_truths.append(ground_truths)

    # Plot histograms
    plt.figure(figsize=(10, 6))
    for i, condition in enumerate(conditions):
        plt.hist(all_predictions[i], bins=20, alpha=0.5, label=f"{condition} (Predictions)")

    plt.xlabel("Predicted Landing Position")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predicted Landing Positions")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "results.png")


if __name__ == "__main__":
    analyze_results()