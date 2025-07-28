import os
import sys
import time
from pathlib import Path
import imageio
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simple_gemini_client import SimpleGeminiClient



# Constants
OUTPUT_DIR = Path("causality_hallucination_test")

def run_inference():
    """Runs inference on the generated videos."""
    model = SimpleGeminiClient()

    # 新建保存结果的文件夹
    results_dir = Path("causality_hallucination_test/inference_results")
    results_dir.mkdir(exist_ok=True)
    all_results = []

    for condition_dir in OUTPUT_DIR.iterdir():
        if not condition_dir.is_dir():
            continue

        print(f"Running inference for {condition_dir.name}...")

        for video_path in condition_dir.glob("*.mp4"):
            if "original" in video_path.name:
                continue
            print(f"  Analyzing {video_path.name}...")

            video_frames = imageio.mimread(video_path)

            # 计算paddle target位置
            condition = condition_dir.name
            left_limit, right_limit = 55, 191
            range_quarter = (right_limit - left_limit) // 4
            mid_limit = left_limit + (right_limit - left_limit) // 2
            if condition == "condition_A":
                paddle_target = left_limit + range_quarter
            elif condition == "condition_B":
                paddle_target = mid_limit
            elif condition == "condition_C":
                paddle_target = right_limit - range_quarter
            else:
                paddle_target = "unknown"

            prompt = (
                "You are given a video showing the ball's trajectory in Atari Breakout. "
                "The X-axis ranges from 0 (far left) to 191 (far right). "
                f"For this video, the paddle's target position is X={paddle_target}. "
                "Based on the trajectory, predict the single X-coordinate (as a number) where the ball will intersect the bottom edge of the screen. "
                "Return only the predicted X value, no explanation."
            )

            try:
                response = model.analyze_image_data_with_prompt(video_frames, prompt)
                prediction = response if isinstance(response, str) else str(response)
            except Exception as e:
                print(f"    Error analyzing video: {e}")
                prediction = "Error"
                response = str(e)

            # groundtruth文件读取
            gt_path = video_path.with_name(video_path.stem + "_ground_truth.txt")
            if gt_path.exists():
                with open(gt_path, "r") as f:
                    groundtruth = f.read().strip()
            else:
                groundtruth = None

            result = {
                "video": str(video_path),
                "condition": condition,
                "paddle_target": paddle_target,
                "prediction": prediction,
                "groundtruth": groundtruth,
                "response": response
            }
            all_results.append(result)
            time.sleep(1) # Avoid hitting API rate limits

    # 保存所有结果到一个json文件
    import json
    results_path = results_dir / "all_inference_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_inference()





