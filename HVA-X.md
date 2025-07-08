Here's the final version of the algorithm, structured as a technical document for implementation.

-----

## **Algorithm: Multi-Pass Video Analysis for RL Agent Summarization**

### **1. Objective**

To programmatically generate a comprehensive, qualitative analysis of a Reinforcement Learning agent's behavior in the game of Breakout. The algorithm uses a multi-pass video analysis approach with a large language model (LLM) to produce a final report detailing the agent's strategies, strengths, and weaknesses.

-----

### **2. Required Inputs**

  * **`rl_agent_model`**: The trained Reinforcement Learning agent capable of playing the game.
  * **`game_environment`**: The Breakout game environment that the agent can interact with.
  * **`gemini_api_key`**: Credentials for accessing the video understanding and text generation API.

-----

### **3. Algorithm Steps**

The algorithm is divided into three primary phases: Sampling, Individual Analysis, and Synthesis.

#### **Phase 1: Trajectory Sampling and Stratification**

The goal of this phase is to collect a representative set of gameplay videos across the agent's full performance spectrum.

1.  **Data Collection**:

      * Run the `rl_agent_model` in the `game_environment` for a large number of episodes (N=100 is recommended).
      * For each episode, save the complete gameplay video and record the final score. Store these as a list of `(video_path, score)` tuples.

2.  **Stratification**:

      * Sort the collected trajectories in ascending order based on their scores.
      * Partition the sorted list into three strata:
          * **`low_tier`**: The bottom 10% of trajectories.
          * **`mid_tier`**: The middle 10% of trajectories (e.g., from the 45th to 55th percentile).
          * **`high_tier`**: The top 10% of trajectories.

3.  **Sampling**:

      * Randomly select 3 to 5 trajectories from each stratum (`low_tier`, `mid_tier`, `high_tier`). This results in a final sample set of 9-15 videos for analysis.

#### **Phase 2: Individual Trajectory Analysis (Two-Pass Method)**

For each video sampled in Phase 1, perform the following two-pass analysis to generate a detailed, context-aware summary.

1.  **Pass 2a: Event Identification**

      * Take a video from the sample set.
      * Make an API call to the video understanding model using the **"Event Detection Prompt"**. The prompt should ask the model to return only a list of timestamps and brief descriptions for key strategic or erroneous moments.
      * Store the resulting list of event timestamps (e.g., `[{timestamp: "0:25-0:31", event: "..."}]`).

2.  **Pass 2b: Guided Analysis**

      * Use the *same* video again.
      * Make a second API call to the video understanding model. This time, use the **"Guided Analysis Prompt"**.
      * This prompt is more detailed, asking for a full strategic breakdown. Crucially, it includes the event timestamps generated in Pass 2a as a "Key Moments for Focused Analysis" section, directing the model's attention.
      * Store the detailed text summary returned by this call, making sure to label it with its performance tier (e.g., `high_tier_summary_1`).

Repeat this two-pass process for all 9-15 sampled videos.

#### **Phase 3: Meta-Summary Synthesis**

This final phase combines all the individual analyses into one cohesive, high-level report.

1.  **Aggregation**:

      * Concatenate all the stored individual summaries (`high_tier_summary_1`, `low_tier_summary_1`, etc.) into a single block of text. Clearly label which summary corresponds to which performance tier within this text block.

2.  **Synthesis**:

      * Make a final API call to a text generation model.
      * Use the **"Synthesis Prompt"**, providing the aggregated text block as the main context.
      * This prompt instructs the model to act as a lead analyst, comparing and contrasting the behaviors described in the input summaries to identify overarching themes, performance differentiators, and common failure modes.

3.  **Final Output**:

      * The text returned from this final API call is the complete, final analysis of the RL agent. This document can be saved or displayed to the end-user.

-----

### **4. Pseudocode**

```python
# Define Prompts (as string templates)
EVENT_DETECTION_PROMPT = "You are an AI assistant... return a list of timestamps..."
GUIDED_ANALYSIS_PROMPT_TEMPLATE = "You are an expert RL analyst... Pay special attention to: {key_events}..."
SYNTHESIS_PROMPT = "You are a lead RL analyst... Synthesize these reports: {all_summaries}..."

function main():
    # Phase 1: Trajectory Sampling
    all_trajectories = run_agent_and_collect_trajectories(num_episodes=100)
    sampled_videos = stratify_and_sample(all_trajectories, num_samples_per_tier=3)
    
    # Phase 2: Individual Trajectory Analysis
    all_individual_analyses = {}
    for tier, videos in sampled_videos.items():
        all_individual_analyses[tier] = []
        for video_path in videos:
            # Pass 2a
            key_events = get_key_events(video_path, EVENT_DETECTION_PROMPT)
            
            # Pass 2b
            prompt = GUIDED_ANALYSIS_PROMPT_TEMPLATE.format(key_events=key_events)
            detailed_summary = get_guided_analysis(video_path, prompt)
            all_individual_analyses[tier].append(detailed_summary)

    # Phase 3: Meta-Summary Synthesis
    final_report = generate_final_summary(all_individual_analyses, SYNTHESIS_PROMPT)
    
    # Final Output
    save_report(final_report, "rl_agent_analysis.md")
    print("Algorithm complete. Analysis saved.")

# --- Helper Functions ---

function run_agent_and_collect_trajectories(num_episodes):
    # Simulates running the agent and saving video/score for each episode.
    # Returns a list of tuples: [(video_path_1, score_1), ...]
    trajectories = []
    for i in range(num_episodes):
        # ... run simulation ...
        # trajectories.append((path, score))
    return trajectories

function stratify_and_sample(trajectories, num_samples_per_tier):
    # Sorts by score, partitions into low/mid/high tiers, and samples from each.
    # Returns a dictionary: {"low_tier": [path1, path2], "high_tier": [...]}
    # ... logic for sorting and sampling ...
    return sampled_videos

function get_key_events(video_path, prompt):
    # Makes the first API call to the video model for event detection.
    # Returns the list of timestamped events.
    # ... API call logic ...
    return events_list

function get_guided_analysis(video_path, prompt):
    # Makes the second API call to the video model for the main analysis.
    # Returns the detailed text summary.
    # ... API call logic ...
    return summary_text

function generate_final_summary(analyses_dict, prompt):
    # Aggregates all individual summaries into a single text block.
    # Makes the final API call to the text model for synthesis.
    # Returns the final report.
    # ... aggregation and final API call logic ...
    return final_report_text
```