from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# It's recommended to set up your API key as an environment variable
# or use another secure method.
client = genai.Client() 

# Define the unified labels we designed.
labels = [
    # Positive Labels
    "Deliberate & Consistent Strategy",
    "Advanced Offensive Execution",
    "High-Efficiency State",
    # Neutral Labels
    # Negative Labels
    "Critical Error on Basic Task",
    "Poor Defensive Reaction",
    "Strategic Rigidity / Failure to Adapt",
    "Inconsistent or Aimless Play",
]

# Extract representative sentences from both agent reports.
sentences_to_test = [
    # From the "Bad" Agent Report
    "It often fails to return the simplest, slowest serve in another.",
    "It shows no consistent high-level strategy at all, focusing only on basic, often unsuccessful, returns.",
    # From the "Good" Agent Report
    "It deliberately carves a channel on one side of the bricks to send the ball behind the wall.",
    "It possesses exceptional paddle control for offensive strikes.",
    "Its defensive responsiveness is poor."

]

# Combine all texts for a single API call to be efficient.
all_texts = labels + sentences_to_test

# Get embeddings for all texts.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=all_texts,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
).embeddings

# Separate the embeddings back into labels and sentences.
embeddings_matrix = np.array([e.values for e in result])
num_labels = len(labels)
embeddings_labels = embeddings_matrix[:num_labels]
embeddings_sentences = embeddings_matrix[num_labels:]

# Calculate cosine similarity between each sentence and each label.
similarity_matrix = cosine_similarity(embeddings_sentences, embeddings_labels)

# Create a pandas DataFrame for easy viewing and saving.
df = pd.DataFrame(similarity_matrix, index=sentences_to_test, columns=labels)

# --- New lines to save the output ---

# Option 1: Save as a CSV file (best for importing into Excel/spreadsheets)
df.to_csv('similarity_report.csv')

# Option 2: Save as a Markdown table (best for reports and easy viewing)
df.to_markdown('similarity_report.md')

print("Successfully saved the analysis to 'similarity_report.csv' and 'similarity_report.md'")