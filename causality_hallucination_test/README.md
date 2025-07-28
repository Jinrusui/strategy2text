# Causal Hallucination Experiment

**Objective:**

Test for the presence of "causal hallucination" in the Gemini API.

**Core Hypothesis:**

An ideal, hallucination-free model should predict the ball's landing position based solely on its trajectory, uninfluenced by the position of a static, irrelevant paddle.

**Experimental Procedure:**

**Step 1: Setup Conditions**

Three independent experimental groups are created, with the only difference being the paddle's position:

*   **Condition A:** Paddle is fixed on the left side throughout the trial.
*   **Condition B:** Paddle is fixed in the middle throughout the trial.
*   **Condition C:** Paddle is fixed on the right side throughout the trial.

**Step 2: Data Sampling**

For each condition (e.g., Condition A - paddle on the left), the following loop is executed 100 times to generate 100 independent samples:

1.  **Environment Initialization (`env.reset()`):**
    *   At the start of the game, randomly remove a few bricks (this adds valuable richness to the scenes).
    *   Fix the paddle to the position specified for Condition A via RAM modification.

2.  **Action Execution:**
    *   Execute the "serve" action to launch the ball at a random initial angle.

3.  **Video Recording and Data Logging:**
    *   Begin recording video frames.
    *   Let the game run with no operations (NOOP) for a short period (e.g., 30-40 frames) to ensure the ball has established a clear trajectory.
    *   Stop video recording *before* the ball hits the paddle or goes off the bottom of the screen. This results in a short video clip for prediction.
    *   **[KEY]:** Although video recording has stopped, allow the game environment to continue running in the background until the ball actually crosses the bottom boundary. Record that precise X-coordinate as the "ground truth" landing position for this sample.

**Step 3: Model Prediction**

*   Input each generated short video clip into your Video Understanding API.
*   Use a precise prompt for the query:

    > "Based on the ball's trajectory in the video, predict the X-coordinate where the ball will land at the bottom of the screen with a single number."

*   Record the predicted landing position returned by the model.

**Step 4: Result Analysis**

1.  **Data Organization:** You should now have three datasets, each containing 100 predicted landing positions and 100 ground truth landing positions.

2.  **Plotting:**
    *   For each condition (A, B, and C), create a frequency distribution plot (histogram) of the predicted landing positions.
    *   Overlay these three plots on the same coordinate system for comparison.

3.  **Verification:**
    *   **Hallucination Test:** Compare the shapes of the three distribution plots.
        *   If the distribution curves are nearly identical, it indicates the model is free of this causal hallucination.
        *   If the peaks of the curves shift in correspondence with the paddle's position (left, middle, right), it provides strong evidence of a significant causal hallucination.
    *   **Accuracy Test (Optional):** You can also calculate the model's prediction accuracy (e.g., Mean Absolute Error - MAE) by comparing the predicted and ground truth landing positions. This primarily evaluates its predictive capability, while the comparison of the distribution plots is the key to testing for hallucination.