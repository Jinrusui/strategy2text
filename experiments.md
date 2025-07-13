experiments motivation:
1. core hypothesis: text strategy summarization is better than video based summarizarion, which more easy to understand, more coverable, and easy to evaluation ability of agent.

# Experiment Documentation

## Motivation

The core hypothesis of this project is that text-based strategy summarization provides a more effective means of understanding agent behavior compared to video-based summarization. Text summaries are easier to comprehend, offer broader coverage, and facilitate more straightforward evaluation of an agent's abilities.

## Experimental Design

1. **Data Collection**: Gather agent trajectories and outcomes from various environments, saving both video recordings and relevant metadata.
2. **Video Analysis**: Use tools like Grad-CAM to visualize important frames and actions in the agent's decision process.
3. **Text Summarization Pipeline**:
   - Extract key events and high-level strategies from the agent's behavior logs.
   - Use natural language processing techniques to generate concise, human-readable summaries.
4. **Evaluation**:
   - Compare the clarity, coverage, and evaluability of text summaries versus video highlights.
   - Conduct user studies or expert reviews to assess which format better communicates agent strategies.
5. **Iteration**: Refine the summarization pipeline based on feedback and experimental results.

## Expected Outcomes

- Demonstrate that text-based summaries are more accessible and informative for understanding agent strategies.
- Provide a framework for automated strategy summarization that can be applied to various reinforcement learning agents.