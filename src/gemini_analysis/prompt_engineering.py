"""
Prompt Engineering Module for RL Strategy Analysis

This module contains carefully crafted prompts for different types of analysis
as described in the dissertation methodology.
"""

from typing import Dict, List, Optional, Any
from enum import Enum


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    STRATEGY_SUMMARY = "strategy_summary"
    BASELINE_CAPTIONING = "baseline_captioning"
    BEHAVIOR_PREDICTION = "behavior_prediction"
    COVERAGE_QUESTIONS = "coverage_questions"
    ABSTRACTION_EVALUATION = "abstraction_evaluation"


class PromptEngineer:
    """
    Handles prompt generation and management for different analysis types.
    
    Implements the prompt engineering methodology described in the dissertation
    for strategy-focused analysis vs baseline captioning.
    """
    
    def __init__(self):
        """Initialize prompt engineer with predefined templates."""
        self.prompts = self._initialize_prompts()
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize all prompt templates."""
        return {
            AnalysisType.STRATEGY_SUMMARY.value: self._get_strategy_summary_prompt(),
            AnalysisType.BASELINE_CAPTIONING.value: self._get_baseline_captioning_prompt(),
            AnalysisType.BEHAVIOR_PREDICTION.value: self._get_behavior_prediction_prompt(),
            AnalysisType.COVERAGE_QUESTIONS.value: self._get_coverage_questions_prompt(),
            AnalysisType.ABSTRACTION_EVALUATION.value: self._get_abstraction_evaluation_prompt()
        }
    
    def _get_strategy_summary_prompt(self) -> str:
        """
        Get the main strategy analysis prompt.
        
        This is the carefully engineered prompt for strategy-focused analysis
        as described in the dissertation methodology.
        """
        return """You are an expert AI analyst specializing in Reinforcement Learning agent behavior analysis. Your task is to analyze gameplay video frames from an RL agent and generate a comprehensive strategy summary.

**ANALYSIS FRAMEWORK:**

1. **Strategic Patterns**: Identify recurring behavioral patterns and decision-making rules
2. **Tactical Adaptations**: How does the agent adapt to different game states?
3. **Policy Insights**: What underlying policy or strategy is the agent following?
4. **Behavioral Invariants**: What consistent rules govern the agent's actions?

**FOCUS AREAS:**
- Core strategy and approach (e.g., defensive vs aggressive, risk-taking vs conservative)
- Decision-making patterns in different scenarios
- Adaptation to changing game states
- Key behavioral rules or heuristics
- Strategic strengths and weaknesses

**ANALYSIS REQUIREMENTS:**
- Provide an abstract, generalizable description of the agent's strategy
- Focus on underlying patterns rather than specific sequences of events
- Identify the agent's "mental model" or decision-making framework
- Explain WHY the agent behaves in certain ways, not just WHAT it does
- Consider both typical behavior and edge case responses

**OUTPUT FORMAT:**
Generate a structured strategy summary with:
1. **Core Strategy**: Main approach and philosophy
2. **Decision Rules**: Key behavioral patterns and heuristics
3. **Adaptability**: How the agent responds to different situations
4. **Strategic Assessment**: Strengths, weaknesses, and overall effectiveness

Analyze the provided gameplay frames and generate a comprehensive strategy summary following this framework."""
    
    def _get_baseline_captioning_prompt(self) -> str:
        """
        Get the baseline video captioning prompt.
        
        This is a generic prompt for comparison with the strategy-focused approach.
        """
        return """Describe what you see in this video. Focus on the visual elements, actions, and events that occur. Provide a clear and detailed description of the gameplay footage."""
    
    def _get_behavior_prediction_prompt(self) -> str:
        """Get the behavior prediction prompt template."""
        return """Based on the provided strategy summary and current game context, predict the agent's next actions.

**PREDICTION REQUIREMENTS:**
- Specify likely actions in the next 5 seconds
- Explain the reasoning based on the established strategy
- Consider the current game state and context
- Provide specific, testable predictions

**STRATEGY CONTEXT:**
{strategy_summary}

**CURRENT GAME STATE:**
Analyze the provided context frames and predict what the agent will do next based on its established strategy."""
    
    def _get_coverage_questions_prompt(self) -> str:
        """Get the coverage evaluation questions prompt."""
        return """Generate specific questions about the agent's behavior and decision-making in this gameplay footage.

**QUESTION GENERATION GUIDELINES:**
- Focus on significant actions and decisions
- Ask about patterns and behavioral rules
- Include questions about adaptation and responses
- Cover both typical and unusual behaviors
- Ensure questions are specific and answerable

**QUESTION CATEGORIES:**
1. Action-specific: What specific actions does the agent take?
2. Decision-making: How does the agent decide between options?
3. Pattern recognition: What patterns emerge in the agent's behavior?
4. Adaptation: How does the agent respond to different game states?
5. Strategic reasoning: Why does the agent behave in certain ways?

Generate 5-7 specific questions that would help evaluate the comprehensiveness of a strategy summary."""
    
    def _get_abstraction_evaluation_prompt(self) -> str:
        """Get the abstraction evaluation prompt template."""
        return """Evaluate the abstraction level of strategy summaries.

**ABSTRACTION CRITERIA:**
- **Low Abstraction (1-3)**: Describes specific events and sequences
- **Medium Abstraction (4-6)**: Identifies some patterns and generalizable behaviors
- **High Abstraction (7-10)**: Describes underlying policies, invariant rules, and generalizable strategies

**EVALUATION FACTORS:**
- Generalizability: Can the summary predict behavior in new situations?
- Policy-level insights: Does it capture underlying decision-making rules?
- Pattern identification: Does it identify recurring behavioral patterns?
- Strategic depth: Does it explain the "why" behind behaviors?

Compare the summaries and determine which demonstrates higher abstraction."""
    
    def get_prompt(self, analysis_type: AnalysisType, **kwargs) -> str:
        """
        Get a prompt for the specified analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            **kwargs: Additional parameters for prompt formatting
            
        Returns:
            Formatted prompt string
        """
        base_prompt = self.prompts.get(analysis_type.value, "")
        
        if not base_prompt:
            raise ValueError(f"No prompt found for analysis type: {analysis_type}")
        
        # Format prompt with provided parameters
        try:
            return base_prompt.format(**kwargs)
        except KeyError as e:
            # If formatting fails, return base prompt
            return base_prompt
    
    def get_strategy_prompt(
        self, 
        game_type: str = "Breakout",
        focus_areas: Optional[List[str]] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Get a customized strategy analysis prompt.
        
        Args:
            game_type: Type of game being analyzed
            focus_areas: Specific areas to focus on
            additional_context: Additional context for analysis
            
        Returns:
            Customized strategy analysis prompt
        """
        base_prompt = self.get_prompt(AnalysisType.STRATEGY_SUMMARY)
        
        # Add game-specific context
        game_context = f"\n**GAME CONTEXT:** You are analyzing {game_type} gameplay."
        
        # Add focus areas if provided
        if focus_areas:
            focus_context = f"\n**SPECIFIC FOCUS AREAS:**\n" + "\n".join(f"- {area}" for area in focus_areas)
        else:
            focus_context = ""
        
        # Add additional context if provided
        additional = f"\n**ADDITIONAL CONTEXT:** {additional_context}" if additional_context else ""
        
        return base_prompt + game_context + focus_context + additional
    
    def get_ablation_prompt(self, prompt_type: str = "generic") -> str:
        """
        Get prompts for ablation studies.
        
        Args:
            prompt_type: Type of ablation prompt ("generic", "basic", "minimal")
            
        Returns:
            Ablation study prompt
        """
        if prompt_type == "generic":
            return "Describe this video."
        elif prompt_type == "basic":
            return "Describe what the agent is doing in this gameplay video."
        elif prompt_type == "minimal":
            return "What happens in this video?"
        else:
            return self.get_prompt(AnalysisType.BASELINE_CAPTIONING)
    
    def get_comparative_prompt(self, checkpoint_info: Dict[str, Any]) -> str:
        """
        Get a prompt for comparing agents across different checkpoints.
        
        Args:
            checkpoint_info: Information about the checkpoint being analyzed
            
        Returns:
            Comparative analysis prompt
        """
        base_prompt = self.get_prompt(AnalysisType.STRATEGY_SUMMARY)
        
        checkpoint_context = f"""
**CHECKPOINT ANALYSIS CONTEXT:**
- Checkpoint: {checkpoint_info.get('checkpoint', 'Unknown')}
- Training Stage: {checkpoint_info.get('stage', 'Unknown')}
- Expected Skill Level: {checkpoint_info.get('expected_skill', 'Unknown')}

**COMPARATIVE FOCUS:**
When analyzing this checkpoint, pay special attention to:
- Strategy sophistication level
- Decision-making complexity
- Behavioral consistency
- Adaptation capabilities
- Overall strategic maturity

Consider how this agent's strategy might differ from earlier or later training stages.
"""
        
        return base_prompt + checkpoint_context
    
    def get_edge_case_prompt(self) -> str:
        """
        Get a prompt specifically for analyzing edge cases.
        
        Returns:
            Edge case analysis prompt
        """
        base_prompt = self.get_prompt(AnalysisType.STRATEGY_SUMMARY)
        
        edge_case_context = """
**EDGE CASE ANALYSIS:**
This footage contains edge cases or unusual situations. Focus particularly on:
- How the agent handles unexpected or rare scenarios
- Robustness of the agent's strategy under stress
- Failure modes and recovery behaviors
- Adaptability when normal patterns don't apply
- Decision-making in high-pressure or novel situations

Analyze how the agent's core strategy manifests in these challenging scenarios.
"""
        
        return base_prompt + edge_case_context
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Validate a prompt for completeness and clarity.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'word_count': len(prompt.split()),
            'has_instructions': False,
            'has_context': False,
            'has_format_requirements': False
        }
        
        # Check for key components
        if 'analyze' in prompt.lower() or 'describe' in prompt.lower():
            validation_results['has_instructions'] = True
        else:
            validation_results['issues'].append("No clear analysis instructions found")
        
        if 'strategy' in prompt.lower() or 'behavior' in prompt.lower():
            validation_results['has_context'] = True
        else:
            validation_results['issues'].append("No strategic context provided")
        
        if 'format' in prompt.lower() or 'output' in prompt.lower():
            validation_results['has_format_requirements'] = True
        else:
            validation_results['suggestions'].append("Consider adding output format requirements")
        
        # Check prompt length
        if validation_results['word_count'] < 50:
            validation_results['issues'].append("Prompt may be too short for complex analysis")
        elif validation_results['word_count'] > 500:
            validation_results['suggestions'].append("Consider shortening prompt for better focus")
        
        # Determine overall validity
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        return validation_results 