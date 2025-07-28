#!/usr/bin/env python3
"""
Enhanced Gemini Embeddings Analysis for HVA-X Agent Reports

This script processes HVA-X agent analysis reports to compute similarity scores
based on positive and negative labels, and generates visualizations.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import json
from datetime import datetime
import argparse

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

class ReportSimilarityAnalyzer:
    def __init__(self, api_client=None):
        """Initialize the analyzer with Gemini API client."""
        self.client = api_client or genai.Client()
        
        # Define positive and negative labels for similarity analysis
        self.positive_labels = [
    "Deliberate & Consistent Strategy",
    "Advanced Offensive Execution",
    "High-Efficiency State",
    "Smart and Adaptive play",
    "Extraordinary Defensive Reaction"
        ]
        
        self.negative_labels = [
    "Critical Error on Basic Task",
    "Poor Defensive Reaction",
    "Strategic Rigidity / Failure to Adapt",
    "Inconsistent or Aimless Play",
    "Loses lifes"
        ]
        
        # Metadata patterns to filter out
        self.metadata_patterns = [
            r'^#+\s+.*',  # Headers
            r'^\*\*Generated:\*\*.*',  # Generated timestamp
            r'^\*\*Algorithm:\*\*.*',  # Algorithm info
            r'^\*\*Phase:\*\*.*',  # Phase info
            r'^-\s+\*\*.*\*\*:.*',  # Bullet points with bold labels
            r'^###\s+Tier Breakdown',  # Tier breakdown section
            r'^---+$',  # Horizontal rules
            r'^\s*$',  # Empty lines
            r'^\d+\.\s+\*\*.*\*\*$',  # Section headers like "1. **Executive Summary**"
            r'^\*\s+\*\*.*\*\*:',  # Bullet points starting analysis sections
        ]
        
    def extract_checkpoint_steps(self, filename: str) -> int:
        """Extract checkpoint steps from filename."""
        match = re.search(r'steps_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    def filter_content_sentences(self, text: str) -> List[str]:
        """Filter out metadata and structural elements, return content sentences."""
        print(f"  Starting content filtering...")
        
        # First, aggressively remove all structural markers and metadata
        # Remove metadata blocks entirely
        text = re.sub(r'\*\*Generated:\*\*.*?\n', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*Algorithm:\*\*.*?\n', '', text)
        text = re.sub(r'\*\*Phase:\*\*.*?\n', '', text)
        
        # Remove entire analysis summary sections
        text = re.sub(r'## Analysis Summary.*?(?=##|\n### |\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'### Tier Breakdown.*?(?=##|\n### |\Z)', '', text, flags=re.DOTALL)
        
        # Remove section headers but be more precise
        text = re.sub(r'^#+\s+\d*\.?\s*[^*\n]*$', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove bullet point metadata (like "- **Input Analyses:** 5")
        text = re.sub(r'^-\s*\*\*[^*]+\*\*:\s*\d+.*$', '', text, flags=re.MULTILINE)
        
        # Remove standalone section labels and bold subsection headers
        text = re.sub(r'^(?:\d+\.\s*)?(?:\*\s*)?(?:\*\*)?(?:Executive Summary|Strategic Analysis|Tactical Skill Assessment|Performance Differentiators|Failure Mode Analysis)(?:\*\*)?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(?:\*\s*)?(?:\*\*)?(?:Primary Strategy|Strategy Consistency|Adaptation|Offensive Skills|Defensive Skills|Consistency|High Performance|Low Performance|Critical Moments|Predictable Failures|Situational Failures|Recovery Patterns)(?:\*\*)?\s*:?\s*$', '', text, flags=re.MULTILINE)
        
        # Remove intro/outro boilerplate
        text = re.sub(r'Here is the comprehensive.*?(?=\n\n|\*\*\*)', '', text, flags=re.DOTALL)
        text = re.sub(r'synthesizing the provided analyses.*?(?=\n\n)', '', text, flags=re.DOTALL)
        
        # Clean up bold markers that are concatenated with content
        # Handle patterns like "**Primary Strategy**: content" -> just "content"
        text = re.sub(r'-\s*\*\*[^*]+\*\*:\s*', '', text)
        text = re.sub(r'\*\s*\*\*[^*]+\*\*:\s*', '', text)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        
        # Split into paragraphs and process
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        # Filter paragraphs to keep only substantial analytical content
        analytical_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 50:  # Reduced from 80 to keep more content
                continue
                
            # Skip paragraphs that start with structural indicators (more permissive)
            if re.match(r'^(?:\d+\.\s*)?(?:\*\s*)?(?:Here is|This (?:script|report|analysis|synthesis))', para, re.IGNORECASE):
                continue
                
            # Must contain key analytical indicators (more permissive)
            analytical_indicators = [
                r'\bthe agent\b', r'\bagent\s+', r'\bstrategy\b', r'\bperformance\b', 
                r'\bskill\b', r'\btactical\b', r'\bdefensive\b', r'\boffensive\b',
                r'\bsuccessful\b', r'\bfailure\b', r'\beffective\b', r'\bconsistent\b',
                r'\bseed\d+\b', r'\btunneling\b', r'\bclearing\b', r'\bbreakout\b', 
                r'\bbrick\b', r'\bpaddle\b', r'\bball\b', r'\bgameplay\b'
            ]
            
            if any(re.search(pattern, para, re.IGNORECASE) for pattern in analytical_indicators):
                analytical_paragraphs.append(para)
        
        print(f"  Kept {len(analytical_paragraphs)} analytical paragraphs")
        
        # Join paragraphs and split into sentences
        full_text = ' '.join(analytical_paragraphs)
        
        # Protect certain patterns during sentence splitting
        full_text = re.sub(r'(\d{2}:\d{2}-\d{2}:\d{2})', r'<TIMESTAMP>\1</TIMESTAMP>', full_text)
        full_text = re.sub(r'(e\.g\.)', r'<ABBREV>eg</ABBREV>', full_text)
        full_text = re.sub(r'(`seed\d+`)', r'<SEED>\1</SEED>', full_text)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+(?=\s+[A-Z]|\s*$)', full_text)
        
        # Filter sentences with moderate criteria
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Restore protected patterns
            sentence = re.sub(r'<TIMESTAMP>(.*?)</TIMESTAMP>', r'\1', sentence)
            sentence = re.sub(r'<ABBREV>eg</ABBREV>', r'e.g.', sentence)
            sentence = re.sub(r'<SEED>(.*?)</SEED>', r'\1', sentence)
            
            # Skip very short sentences
            if not sentence or len(sentence) < 15:  # Reduced from 20
                continue
            
            # Filter out clear structural remnants (but be less aggressive)
            structural_patterns = [
                r'^(?:\d+\.\s*)?(?:\*\s*)?(?:Executive Summary|Strategic Analysis|Tactical Skill Assessment|Performance Differentiators|Failure Mode Analysis)(?:\s*:|\s*$)',
                r'^(?:\d+\.\s*)?(?:\*\s*)?(?:Primary Strategy|Strategy Consistency|Adaptation|Offensive Skills|Defensive Skills)(?:\s*:|\s*$)',
                r'^(?:\d+\.\s*)?(?:\*\s*)?(?:High Performance|Low Performance|Critical Moments|Predictable Failures|Situational Failures|Recovery Patterns)(?:\s*:|\s*$)',
                r'^Here is (?:the|a) (?:comprehensive|detailed|complete)',
                r'^This (?:synthesis|report|analysis) (?:provides|examines|presents)',
                r'synthesizing the provided analyses',
                r'with specific video references',
                r'^-\s*(?:All Videos|Input Analyses|Failed Analyses)',
                r'^\*\*[^*]+\*\*:?\s*$',  # Standalone bold labels only
                r'^[\d\s\.\-\*\(\)IVX]+$',  # Just numbers and punctuation
                r'^[A-Za-z\s\-]*:\s*$',  # Pure labels ending with colon only
                r'^(?:Analysis Summary|Tier Breakdown)\s*$',  # Standalone section titles only
                r'^#+\s*$',  # Section markers only
                r'^rally\s*$',  # Just the word "rally"
            ]
            
            # Check if sentence matches any structural pattern
            is_structural = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in structural_patterns)
            if is_structural:
                continue
            
            # More permissive content validation
            # Keep sentences that have analytical content OR game-specific content
            analytical_content_patterns = [
                r'\bthe agent\b',
                r'\bagent\s+(?:demonstrates|shows|displays|exhibits|fails|performs|struggles|succeeds|attempts|possesses|can|cannot|is|does)',
                r'\b(?:strategy|performance|skill|ability|competence|tactical|defensive|offensive)\b',
                r'\b(?:successful|failure|effective|ineffective|consistent|inconsistent)\b.*\b(?:execution|implementation|approach|behavior|pattern)\b',
                r'\bseed\d+\b.*\d{2}:\d{2}',  # Video references with timestamps
                r'\b(?:tunneling|clearing|defensive|offensive|tactical|strategic)\b.*\b(?:play|behavior|action|response|approach)\b',
                r'\b(?:leads to|results in|causes|triggers|demonstrates|indicates|suggests|reveals|shows)\b',
                r'\b(?:breakout|brick|paddle|ball|game|gameplay|rally|miss|hit|score)\b',
                r'this\b.*\b(?:pattern|behavior|approach|strategy|weakness|strength|flaw|skill)\b',
            ]
            
            # Keep sentences with analytical content
            has_analytical_content = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in analytical_content_patterns)
            
            # Additional check: reject obvious fragments or very short content
            is_substantial = (
                sentence.count(' ') >= 5 and  # At least 6 words (reduced from 8)
                not re.match(r'^(?:-|\*|\d+\.)\s', sentence) and  # Not a list item
                not sentence.endswith(':')  # Not ending with colon (incomplete)
            )
            
            if has_analytical_content and is_substantial:
                # Final cleanup
                sentence = re.sub(r'\s+', ' ', sentence)  # Normalize whitespace
                sentence = sentence.strip()
                if sentence and not sentence.endswith(('.', '!', '?')):
                    sentence += '.'  # Ensure proper sentence ending
                filtered_sentences.append(sentence)
        
        print(f"  Final filtered sentences: {len(filtered_sentences)}")
        return filtered_sentences
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using Gemini."""
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            return np.array([e.values for e in result.embeddings])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return np.array([])
    
    def compute_sentence_score(self, sentence_embedding: np.ndarray, 
                             positive_embeddings: np.ndarray, 
                             negative_embeddings: np.ndarray) -> float:
        """Compute score for a single sentence."""
        # Calculate similarities
        pos_similarities = cosine_similarity([sentence_embedding], positive_embeddings)[0]
        neg_similarities = cosine_similarity([sentence_embedding], negative_embeddings)[0]
        
        # Score = sum of positive similarities - sum of negative similarities
        score = np.sum(pos_similarities) - np.sum(neg_similarities)
        return score
    
    def analyze_report(self, report_path: str) -> Dict:
        """Analyze a single report and return scoring results."""
        print(f"Analyzing report: {report_path}")
        
        # Read report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract checkpoint steps
        checkpoint_steps = self.extract_checkpoint_steps(report_path)
        
        # Filter content sentences
        sentences = self.filter_content_sentences(content)
        print(f"  Found {len(sentences)} content sentences")
        
        if not sentences:
            return {
                'checkpoint_steps': checkpoint_steps,
                'normalized_score': 0,
                'sentence_count': 0,
                'sentence_scores': []
            }
        
        # Get embeddings for labels (do this once)
        all_labels = self.positive_labels + self.negative_labels
        label_embeddings = self.get_embeddings(all_labels)
        
        if label_embeddings.size == 0:
            print("  Error getting label embeddings")
            return {
                'checkpoint_steps': checkpoint_steps,
                'normalized_score': 0,
                'sentence_count': 0,
                'sentence_scores': []
            }
        
        positive_embeddings = label_embeddings[:len(self.positive_labels)]
        negative_embeddings = label_embeddings[len(self.positive_labels):]
        
        # Process sentences in batches to avoid API limits
        batch_size = 10
        sentence_scores = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_embeddings = self.get_embeddings(batch_sentences)
            
            if batch_embeddings.size == 0:
                print(f"  Error getting embeddings for batch {i//batch_size + 1}")
                continue
            
            for j, sentence_embedding in enumerate(batch_embeddings):
                score = self.compute_sentence_score(
                    sentence_embedding, positive_embeddings, negative_embeddings
                )
                sentence_scores.append({
                    'sentence': batch_sentences[j],
                    'score': score
                })
        
        # Calculate normalized score: reduce influence of sentence count
        total_score = sum(s['score'] for s in sentence_scores)
        # Use square root normalization to dampen the effect of having many sentences
        normalized_score = total_score / (len(sentence_scores) ** 0.5) if sentence_scores else 0
        
        return {
            'checkpoint_steps': checkpoint_steps,
            'normalized_score': normalized_score,
            'sentence_count': len(sentence_scores),
            'sentence_scores': sentence_scores
        }
    
    def analyze_all_reports(self, reports_pattern: str) -> pd.DataFrame:
        """Analyze all reports matching the pattern."""
        report_files = glob.glob(reports_pattern)
        print(f"Found {len(report_files)} report files")
        
        results = []
        for report_file in sorted(report_files):
            result = self.analyze_report(report_file)
            result['report_file'] = os.path.basename(report_file)
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        return df.sort_values('checkpoint_steps')
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "."):
        """Create visualizations of the results, including average score overlay and improved x-axis."""
        import json
        os.makedirs(output_dir, exist_ok=True)

        # Load average scores
        avg_scores_path = os.path.join("checkpoint_analysis_results", "average_scores.json")
        if os.path.exists(avg_scores_path):
            with open(avg_scores_path, 'r') as f:
                avg_scores = json.load(f)
        else:
            avg_scores = {}

        # Prepare data for plotting
        steps = df['checkpoint_steps'].astype(int).tolist()
        norm_scores = df['normalized_score'].tolist()
        # Match average scores to steps (if available)
        avg_score_list = [avg_scores.get(f"steps_{int(s):08d}", None) for s in steps]

        # Create figure with dual y-axes using a more modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('white')

        # Define colors
        color1 = '#E74C3C'  # Modern red
        color2 = '#3498DB'  # Modern blue

        # Plot normalized similarity score on left y-axis
        ax1.set_xlabel('Training Checkpoint Steps', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Normalized Similarity Score', color=color1, fontsize=16, fontweight='bold')
        line1 = ax1.plot(steps, norm_scores, 'o-', linewidth=4, markersize=12, color=color1, 
                        label='Similarity Score', markeredgecolor='white', markeredgewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Create second y-axis for average score
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Score in Game', color=color2, fontsize=16, fontweight='bold')
        
        # Plot average score if available
        valid_steps = []
        valid_avg_scores = []
        if any(x is not None for x in avg_score_list):
            # Filter out None values for plotting
            valid_steps = [s for s, avg in zip(steps, avg_score_list) if avg is not None]
            valid_avg_scores = [avg for avg in avg_score_list if avg is not None]
            line2 = ax2.plot(valid_steps, valid_avg_scores, 's--', linewidth=3, markersize=10, 
                           color=color2, label='Average Score in Game', markeredgecolor='white', markeredgewidth=2)
            ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

        # Improved title
        plt.suptitle('HVA-X Agent Performance Analysis', fontsize=20, fontweight='bold', y=0.95)
        plt.title('Normalized Similarity Score & Average Game Score vs Training Progress', 
                 fontsize=14, style='italic', pad=20)

        # Smart label positioning to avoid overlaps
        def get_label_positions(x_vals, y_vals, offset_base=20):
            """Calculate non-overlapping label positions."""
            positions = []
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                # Alternate above and below, with adjustments for nearby points
                base_offset = offset_base if i % 2 == 0 else -offset_base - 10
                
                # Check for nearby points and adjust
                for j, (prev_x, prev_y) in enumerate(zip(x_vals[:i], y_vals[:i])):
                    if abs(x - prev_x) < (max(x_vals) - min(x_vals)) * 0.1:  # If x-values are close
                        if abs(y - prev_y) < (max(y_vals) - min(y_vals)) * 0.2:  # And y-values are close
                            base_offset *= -1.5  # Flip and increase offset
                            break
                
                positions.append(base_offset)
            return positions

        # Add score annotations for normalized similarity scores with smart positioning
        norm_offsets = get_label_positions(steps, norm_scores, 25)
        for i, (step, score, offset) in enumerate(zip(steps, norm_scores, norm_offsets)):
            ax1.annotate(f'{score:.2f}', 
                        xy=(step, score),
                        xytext=(0, offset), textcoords='offset points', 
                        fontsize=10, ha='center', color=color1, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                                edgecolor=color1, linewidth=1.5),
                        arrowprops=dict(arrowstyle='->', color=color1, alpha=0.7, lw=1))

        # Add score annotations for average scores with smart positioning
        if valid_steps:
            avg_offsets = get_label_positions(valid_steps, valid_avg_scores, -30)
            for step, score, offset in zip(valid_steps, valid_avg_scores, avg_offsets):
                ax2.annotate(f'{score:.0f}', 
                            xy=(step, score),
                            xytext=(0, offset), textcoords='offset points', 
                            fontsize=10, ha='center', color=color2, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                                    edgecolor=color2, linewidth=1.5),
                            arrowprops=dict(arrowstyle='->', color=color2, alpha=0.7, lw=1))

        # Add prominent annotations for min and max points (normalized score)
        max_idx = df['normalized_score'].idxmax()
        min_idx = df['normalized_score'].idxmin()
        
        ax1.annotate(f'Peak Performance\n{df.loc[max_idx, "normalized_score"]:.2f}', 
                    xy=(df.loc[max_idx, 'checkpoint_steps'], df.loc[max_idx, 'normalized_score']),
                    xytext=(30, 30), textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.9, edgecolor='orange'),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        
        ax1.annotate(f'Lowest Performance\n{df.loc[min_idx, "normalized_score"]:.2f}', 
                    xy=(df.loc[min_idx, 'checkpoint_steps'], df.loc[min_idx, 'normalized_score']),
                    xytext=(30, -40), textcoords='offset points', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9, edgecolor='red'),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        # Refine x-axis: show precise checkpoint steps with better formatting
        step_labels = []
        for step in steps:
            if step >= 1000000:
                step_labels.append(f"{step/1000000:.1f}M")
            elif step >= 1000:
                step_labels.append(f"{step/1000:.0f}K")
            else:
                step_labels.append(str(step))
        
        ax1.set_xticks(steps)
        ax1.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')

        # Create a more beautiful legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if valid_steps else ([], [])
        
        # Combine legends with better styling
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, 
                          loc='upper left', fontsize=14, frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.95, edgecolor='gray')
        legend.get_frame().set_facecolor('white')

        # Add subtle background styling
        ax1.set_facecolor('#FAFAFA')
        
        # Improve layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save with high quality
        plt.savefig(os.path.join(output_dir, 'normalized_similarity_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved enhanced visualization to {output_dir}/normalized_similarity_analysis.png")
        
        plt.close('all')  # Close all figures
    
    def save_results(self, df: pd.DataFrame, detailed_results: List[Dict], output_dir: str = "."):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary DataFrame
        summary_file = os.path.join(output_dir, 'similarity_summary.csv')
        df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
        
        # Save detailed results
        detailed_file = os.path.join(output_dir, 'detailed_similarity_results.json')
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"Saved detailed results to {detailed_file}")
        
        # Create a markdown report
        report_file = os.path.join(output_dir, 'similarity_analysis_report.md')
        with open(report_file, 'w') as f:
            f.write("# HVA-X Agent Similarity Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Reports Analyzed:** {len(df)}\n")
            f.write(f"- **Checkpoint Range:** {df['checkpoint_steps'].min():,} - {df['checkpoint_steps'].max():,}\n")
            f.write(f"- **Best Performing Checkpoint:** {df.loc[df['normalized_score'].idxmax(), 'checkpoint_steps']:,} (Score: {df['normalized_score'].max():.2f})\n")
            f.write(f"- **Worst Performing Checkpoint:** {df.loc[df['normalized_score'].idxmin(), 'checkpoint_steps']:,} (Score: {df['normalized_score'].min():.2f})\n\n")
            
            f.write("## Scoring Method\n\n")
            f.write("**Normalized Score:** Total similarity score divided by sqrt(sentence count) to reduce sentence count bias\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Checkpoint Steps | Normalized Score | Sentence Count |\n")
            f.write("|------------------|------------------|----------------|\n")
            for _, row in df.iterrows():
                f.write(f"| {row['checkpoint_steps']:,} | {row['normalized_score']:.2f} | {row['sentence_count']} |\n")
        
        print(f"Saved markdown report to {report_file}")

    def plot_only_mode(self, output_dir: str = "similarity_analysis_results"):
        """Create visualizations from existing results without rerunning similarity analysis."""
        print("Running in plot-only mode...")
        
        # Load existing results
        summary_file = os.path.join(output_dir, 'similarity_summary.csv')
        if not os.path.exists(summary_file):
            print(f"Error: {summary_file} not found. Please run the full analysis first.")
            return
        
        # Read the summary data
        df = pd.read_csv(summary_file)
        print(f"Loaded {len(df)} records from existing results")
        
        # Create visualizations
        self.create_visualizations(df, output_dir)
        print(f"Plotting complete! Enhanced visualization saved to '{output_dir}' directory.")


def main():
    """Main execution function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='HVA-X Report Similarity Analysis')
    parser.add_argument('--plot-only', action='store_true', 
                       help='Only create plots from existing results (skip similarity analysis)')
    parser.add_argument('--output-dir', default='similarity_analysis_results',
                       help='Output directory for results (default: similarity_analysis_results)')
    
    args = parser.parse_args()
    
    print("Starting HVA-X Report Similarity Analysis...")
    
    # Initialize analyzer
    analyzer = ReportSimilarityAnalyzer()
    
    # Check if running in plot-only mode
    if args.plot_only:
        analyzer.plot_only_mode(args.output_dir)
        return
    
    # Define pattern for report files
    reports_pattern = "/mnt/e/Projects/strategy2text/checkpoint_analysis_results/*/phase3_*_report_*.md"
    
    # Analyze all reports
    df = analyzer.analyze_all_reports(reports_pattern)
    
    if df.empty:
        print("No reports found or analyzed successfully.")
        return
    
    print(f"\nAnalysis complete! Processed {len(df)} reports.")
    print("\nSummary Statistics:")
    print(f"Best performing checkpoint: {df.loc[df['normalized_score'].idxmax(), 'checkpoint_steps']:,} (Score: {df['normalized_score'].max():.2f})")
    print(f"Worst performing checkpoint: {df.loc[df['normalized_score'].idxmin(), 'checkpoint_steps']:,} (Score: {df['normalized_score'].min():.2f})")
    
    print(f"\nCheckpoint Ranking (by Normalized Score):")
    ranking = df.nlargest(len(df), 'normalized_score')['checkpoint_steps'].tolist()
    print(f"Ranking: {ranking}")
    
    # Create output directory
    output_dir = args.output_dir
    
    # Save results
    detailed_results = df.to_dict('records')
    analyzer.save_results(df, detailed_results, output_dir)
    
    # Create visualizations
    analyzer.create_visualizations(df, output_dir)
    
    print(f"\nAll results saved to '{output_dir}' directory.")


if __name__ == "__main__":
    main()
