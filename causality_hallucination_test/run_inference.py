import os
import sys
import time
from pathlib import Path
import imageio
from simple_gemini_client import SimpleGeminiClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants
OUTPUT_DIR = Path("causality_hallucination_test")

def run_inference():
    """Runs inference on the generated videos."""
    model = SimpleGeminiClient()

    for condition_dir in OUTPUT_DIR.iterdir():
        if not condition_dir.is_dir():
            continue

        print(f"Running inference for {condition_dir.name}...")

        for video_path in condition_dir.glob("*.mp4"):
            print(f"  Analyzing {video_path.name}...")
            prediction_path = video_path.with_suffix(".txt")
            if prediction_path.exists():
                print(f"    Prediction already exists, skipping.")
                continue

            video_frames = imageio.mimread(video_path)

            prompt = "Based on the ball's trajectory in the video, predict the X-coordinate where the ball will land at the bottom of the screen with a single number."

            try:
                prediction = model.analyze_image_data_with_prompt(video_frames, prompt)
            except Exception as e:
                print(f"    Error analyzing video: {e}")
                prediction = "Error"

            with open(prediction_path, "w") as f:
                f.write(prediction)

            time.sleep(1) # Avoid hitting API rate limits

if __name__ == "__main__":
    run_inference()
