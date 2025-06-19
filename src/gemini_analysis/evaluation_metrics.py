"""
Evaluation Metrics Module for Strategy Analysis

Implements the three key metrics described in the dissertation:
1. Predictive Faithfulness Score (PFS)
2. Coverage Score
3. Abstraction Score
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import re
from collections import Counter


class EvaluationMetrics:
    """
    Implements evaluation metrics for strategy analysis quality assessment.
    
    Provides quantitative measures for faithfulness, coverage, and abstraction
    as described in the dissertation methodology.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize evaluation metrics.
        
        Args:
            embedding_model: SentenceTransformer model for semantic similarity
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def calculate_predictive_faithfulness_score(
        self,
        prediction: str,
        ground_truth: str,
        method: str = "semantic"
    ) -> float:
        """
        Calculate Predictive Faithfulness Score (PFS).
        
        Measures semantic similarity between predicted behavior and actual behavior.
        
        Args:
            prediction: Predicted behavior description
            ground_truth: Actual behavior description
            method: Similarity calculation method ("semantic", "bleu", "cosine")
            
        Returns:
            PFS score between 0 and 1
        """
        if not prediction.strip() or not ground_truth.strip():
            return 0.0
        
        if method == "semantic":
            return self._calculate_semantic_similarity(prediction, ground_truth)
        elif method == "bleu":
            return self._calculate_bleu_score(prediction, ground_truth)
        elif method == "cosine":
            return self._calculate_cosine_similarity(prediction, ground_truth)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings."""
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, similarity)  # Ensure non-negative
    
    def _calculate_bleu_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate BLEU score between prediction and ground truth."""
        # Tokenize texts
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        
        # Calculate BLEU score
        try:
            score = sentence_bleu([ground_truth_tokens], prediction_tokens)
            return score
        except:
            return 0.0
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between text vectors."""
        # Simple word-based vectorization
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Create vocabulary
        vocab = words1.union(words2)
        
        # Create vectors
        vec1 = np.array([1 if word in words1 else 0 for word in vocab])
        vec2 = np.array([1 if word in words2 else 0 for word in vocab])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_coverage_score(
        self,
        strategy_summary: str,
        questions: List[str],
        answers: List[str],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Calculate Coverage Score.
        
        Measures how comprehensively the strategy summary covers agent behaviors.
        
        Args:
            strategy_summary: Generated strategy summary
            questions: List of questions about agent behavior
            answers: List of answers derived from the strategy summary
            threshold: Threshold for considering an answer adequate
            
        Returns:
            Dictionary with coverage metrics
        """
        if not questions or not answers or len(questions) != len(answers):
            return {
                'coverage_score': 0.0,
                'answered_questions': 0,
                'total_questions': len(questions),
                'adequate_answers': 0,
                'answer_quality_scores': []
            }
        
        answer_quality_scores = []
        adequate_answers = 0
        
        for question, answer in zip(questions, answers):
            # Calculate answer quality based on multiple factors
            quality_score = self._evaluate_answer_quality(question, answer, strategy_summary)
            answer_quality_scores.append(quality_score)
            
            if quality_score >= threshold:
                adequate_answers += 1
        
        coverage_score = adequate_answers / len(questions) if questions else 0.0
        
        return {
            'coverage_score': coverage_score,
            'answered_questions': len([a for a in answers if a.strip()]),
            'total_questions': len(questions),
            'adequate_answers': adequate_answers,
            'answer_quality_scores': answer_quality_scores,
            'average_answer_quality': np.mean(answer_quality_scores) if answer_quality_scores else 0.0
        }
    
    def _evaluate_answer_quality(self, question: str, answer: str, strategy_summary: str) -> float:
        """
        Evaluate the quality of an answer to a question.
        
        Args:
            question: The question being answered
            answer: The provided answer
            strategy_summary: The strategy summary used to generate the answer
            
        Returns:
            Quality score between 0 and 1
        """
        if not answer.strip():
            return 0.0
        
        # Check if answer indicates insufficient information
        insufficient_indicators = [
            "not enough information",
            "cannot determine",
            "unclear",
            "not specified",
            "insufficient data"
        ]
        
        if any(indicator in answer.lower() for indicator in insufficient_indicators):
            return 0.2  # Low score for insufficient information
        
        # Calculate relevance to question
        question_relevance = self._calculate_semantic_similarity(question, answer)
        
        # Calculate connection to strategy summary
        strategy_connection = self._calculate_semantic_similarity(strategy_summary, answer)
        
        # Calculate answer completeness (length-based heuristic)
        completeness = min(1.0, len(answer.split()) / 20)  # Normalize to 20 words
        
        # Combine scores
        quality_score = (question_relevance * 0.4 + 
                        strategy_connection * 0.4 + 
                        completeness * 0.2)
        
        return min(1.0, quality_score)
    
    def calculate_abstraction_score(
        self,
        summary: str,
        method: str = "linguistic"
    ) -> Dict[str, Any]:
        """
        Calculate Abstraction Score.
        
        Measures how abstract vs concrete a strategy summary is.
        
        Args:
            summary: Strategy summary to evaluate
            method: Evaluation method ("linguistic", "pattern", "hybrid")
            
        Returns:
            Dictionary with abstraction metrics
        """
        if not summary.strip():
            return {
                'abstraction_score': 0.0,
                'concrete_indicators': 0,
                'abstract_indicators': 0,
                'method': method
            }
        
        if method == "linguistic":
            return self._calculate_linguistic_abstraction(summary)
        elif method == "pattern":
            return self._calculate_pattern_abstraction(summary)
        elif method == "hybrid":
            linguistic_result = self._calculate_linguistic_abstraction(summary)
            pattern_result = self._calculate_pattern_abstraction(summary)
            
            # Combine scores
            combined_score = (linguistic_result['abstraction_score'] + 
                            pattern_result['abstraction_score']) / 2
            
            return {
                'abstraction_score': combined_score,
                'linguistic_score': linguistic_result['abstraction_score'],
                'pattern_score': pattern_result['abstraction_score'],
                'concrete_indicators': linguistic_result['concrete_indicators'],
                'abstract_indicators': linguistic_result['abstract_indicators'],
                'method': method
            }
        else:
            raise ValueError(f"Unknown abstraction method: {method}")
    
    def _calculate_linguistic_abstraction(self, summary: str) -> Dict[str, Any]:
        """Calculate abstraction based on linguistic features."""
        # Define concrete and abstract indicators
        concrete_indicators = [
            # Specific actions
            r'\bmove[sd]?\b', r'\bhit[s]?\b', r'\bjump[s]?\b', r'\bclick[s]?\b',
            # Specific sequences
            r'\bfirst\b', r'\bthen\b', r'\bnext\b', r'\bafter\b',
            # Specific numbers/times
            r'\b\d+\s*(second|minute|time|frame)[s]?\b',
            # Specific locations
            r'\bleft side\b', r'\bright side\b', r'\btop\b', r'\bbottom\b'
        ]
        
        abstract_indicators = [
            # Strategic concepts
            r'\bstrategy\b', r'\bapproach\b', r'\bpolicy\b', r'\bmethod\b',
            # Patterns and rules
            r'\bpattern[s]?\b', r'\brule[s]?\b', r'\bheuristic[s]?\b', r'\bprinciple[s]?\b',
            # General behaviors
            r'\btypically\b', r'\busually\b', r'\bgenerally\b', r'\btends to\b',
            # Abstract reasoning
            r'\bbecause\b', r'\bin order to\b', r'\bso that\b', r'\bwhen.*then\b'
        ]
        
        # Count indicators
        concrete_count = sum(len(re.findall(pattern, summary.lower())) 
                           for pattern in concrete_indicators)
        abstract_count = sum(len(re.findall(pattern, summary.lower())) 
                           for pattern in abstract_indicators)
        
        # Calculate abstraction score
        total_indicators = concrete_count + abstract_count
        if total_indicators == 0:
            abstraction_score = 0.5  # Neutral if no indicators found
        else:
            abstraction_score = abstract_count / total_indicators
        
        return {
            'abstraction_score': abstraction_score,
            'concrete_indicators': concrete_count,
            'abstract_indicators': abstract_count,
            'method': 'linguistic'
        }
    
    def _calculate_pattern_abstraction(self, summary: str) -> Dict[str, Any]:
        """Calculate abstraction based on pattern recognition."""
        # Look for pattern-indicating structures
        pattern_indicators = [
            # Conditional statements
            len(re.findall(r'\bif\b.*\bthen\b', summary.lower())),
            # Causal relationships
            len(re.findall(r'\bbecause\b|\bdue to\b|\bresults in\b', summary.lower())),
            # Generalizations
            len(re.findall(r'\balways\b|\bnever\b|\busually\b|\btypically\b', summary.lower())),
            # Strategic terminology
            len(re.findall(r'\bstrategy\b|\btactic[s]?\b|\bapproach\b|\bpolicy\b', summary.lower()))
        ]
        
        # Specific event indicators (concrete)
        specific_indicators = [
            # Temporal specificity
            len(re.findall(r'\bat \d+\b|\bafter \d+\b|\bin \d+ second[s]?\b', summary.lower())),
            # Spatial specificity
            len(re.findall(r'\bat position\b|\bcoordinate[s]?\b|\bpixel[s]?\b', summary.lower())),
            # Sequence specificity
            len(re.findall(r'\bstep \d+\b|\bfirst.*second.*third\b', summary.lower()))
        ]
        
        pattern_score = sum(pattern_indicators)
        specific_score = sum(specific_indicators)
        
        # Calculate abstraction ratio
        total_score = pattern_score + specific_score
        if total_score == 0:
            abstraction_score = 0.5
        else:
            abstraction_score = pattern_score / total_score
        
        return {
            'abstraction_score': abstraction_score,
            'concrete_indicators': specific_score,
            'abstract_indicators': pattern_score,
            'method': 'pattern'
        }
    
    def compare_summaries(
        self,
        summary1: str,
        summary2: str,
        metrics: List[str] = ["faithfulness", "coverage", "abstraction"]
    ) -> Dict[str, Any]:
        """
        Compare two strategy summaries across multiple metrics.
        
        Args:
            summary1: First strategy summary
            summary2: Second strategy summary
            metrics: List of metrics to compare
            
        Returns:
            Comparison results
        """
        results = {
            'summary1': summary1,
            'summary2': summary2,
            'metrics': {}
        }
        
        if "abstraction" in metrics:
            abs1 = self.calculate_abstraction_score(summary1)
            abs2 = self.calculate_abstraction_score(summary2)
            
            results['metrics']['abstraction'] = {
                'summary1_score': abs1['abstraction_score'],
                'summary2_score': abs2['abstraction_score'],
                'winner': 'summary1' if abs1['abstraction_score'] > abs2['abstraction_score'] else 'summary2',
                'difference': abs(abs1['abstraction_score'] - abs2['abstraction_score'])
            }
        
        # Add semantic similarity between summaries
        if len(metrics) > 1:
            similarity = self._calculate_semantic_similarity(summary1, summary2)
            results['semantic_similarity'] = similarity
        
        return results
    
    def generate_evaluation_report(
        self,
        strategy_summary: str,
        evaluation_data: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            strategy_summary: The strategy summary being evaluated
            evaluation_data: Dictionary containing evaluation results
            
        Returns:
            Formatted evaluation report
        """
        report = f"""
STRATEGY ANALYSIS EVALUATION REPORT
====================================

STRATEGY SUMMARY:
{strategy_summary[:200]}{'...' if len(strategy_summary) > 200 else ''}

EVALUATION METRICS:
"""
        
        # Add PFS results if available
        if 'pfs' in evaluation_data:
            pfs_data = evaluation_data['pfs']
            report += f"""
PREDICTIVE FAITHFULNESS SCORE (PFS):
- Score: {pfs_data.get('score', 'N/A'):.3f}
- Method: {pfs_data.get('method', 'N/A')}
- Interpretation: {'High faithfulness' if pfs_data.get('score', 0) > 0.7 else 'Moderate faithfulness' if pfs_data.get('score', 0) > 0.5 else 'Low faithfulness'}
"""
        
        # Add Coverage results if available
        if 'coverage' in evaluation_data:
            cov_data = evaluation_data['coverage']
            report += f"""
COVERAGE SCORE:
- Overall Score: {cov_data.get('coverage_score', 'N/A'):.3f}
- Questions Answered: {cov_data.get('answered_questions', 'N/A')}/{cov_data.get('total_questions', 'N/A')}
- Adequate Answers: {cov_data.get('adequate_answers', 'N/A')}
- Average Answer Quality: {cov_data.get('average_answer_quality', 'N/A'):.3f}
"""
        
        # Add Abstraction results if available
        if 'abstraction' in evaluation_data:
            abs_data = evaluation_data['abstraction']
            report += f"""
ABSTRACTION SCORE:
- Score: {abs_data.get('abstraction_score', 'N/A'):.3f}
- Abstract Indicators: {abs_data.get('abstract_indicators', 'N/A')}
- Concrete Indicators: {abs_data.get('concrete_indicators', 'N/A')}
- Method: {abs_data.get('method', 'N/A')}
- Level: {'High abstraction' if abs_data.get('abstraction_score', 0) > 0.7 else 'Moderate abstraction' if abs_data.get('abstraction_score', 0) > 0.5 else 'Low abstraction'}
"""
        
        report += "\n" + "="*50
        
        return report 