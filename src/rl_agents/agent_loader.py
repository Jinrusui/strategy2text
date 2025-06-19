"""
Agent loader utility for managing multiple SB3 agents and checkpoints.

Provides utilities for discovering, loading, and organizing trained agents.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

from .sb3_agent import SB3Agent


class AgentLoader:
    """
    Utility for loading and managing multiple SB3 agents.
    
    Supports:
    - Discovering agents in directories
    - Loading agents with metadata
    - Organizing agents by training stage/checkpoint
    - Batch operations on multiple agents
    """
    
    def __init__(self, agents_dir: str = "models/agents"):
        """
        Initialize agent loader.
        
        Args:
            agents_dir: Directory containing agent models
        """
        self.agents_dir = Path(agents_dir)
        self.loaded_agents: Dict[str, SB3Agent] = {}
        self.agent_registry: List[Dict[str, Any]] = []
        
        if self.agents_dir.exists():
            self._discover_agents()
    
    def _discover_agents(self):
        """Discover agent files in the agents directory."""
        print(f"Discovering agents in {self.agents_dir}")
        
        # Common SB3 model file extensions
        model_extensions = ['.zip', '.pkl', '.pt', '.pth']
        
        for model_file in self.agents_dir.rglob('*'):
            if model_file.suffix.lower() in model_extensions:
                agent_info = self._extract_agent_info(model_file)
                if agent_info:
                    self.agent_registry.append(agent_info)
        
        print(f"Discovered {len(self.agent_registry)} agent files")
    
    def _extract_agent_info(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract agent information from file path and name.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Agent information dictionary
        """
        try:
            filename = model_path.stem
            
            # Try to extract information from filename
            # Expected patterns: algorithm_env_checkpoint_timestamp.zip
            # e.g., ppo_breakout_1000000_20231201.zip
            
            agent_info = {
                'path': str(model_path),
                'filename': model_path.name,
                'algorithm': self._extract_algorithm(filename),
                'environment': self._extract_environment(filename),
                'checkpoint': self._extract_checkpoint(filename),
                'timestamp': self._extract_timestamp(filename),
                'file_size': model_path.stat().st_size,
                'created_time': model_path.stat().st_ctime
            }
            
            # Load metadata if available
            metadata_path = model_path.parent / f"{filename}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    agent_info['metadata'] = metadata
            
            return agent_info
            
        except Exception as e:
            print(f"Error extracting info from {model_path}: {e}")
            return None
    
    def _extract_algorithm(self, filename: str) -> Optional[str]:
        """Extract algorithm from filename."""
        filename_lower = filename.lower()
        algorithms = ['ppo', 'a2c', 'dqn', 'sac', 'td3']
        
        for alg in algorithms:
            if alg in filename_lower:
                return alg
        
        return None
    
    def _extract_environment(self, filename: str) -> Optional[str]:
        """Extract environment from filename."""
        # Common environment names
        envs = ['breakout', 'pong', 'spaceinvaders', 'seaquest', 'cartpole', 'lunarlander']
        
        filename_lower = filename.lower()
        for env in envs:
            if env in filename_lower:
                return env
        
        return None
    
    def _extract_checkpoint(self, filename: str) -> Optional[int]:
        """Extract checkpoint/timestep from filename."""
        # Look for patterns like _1000000_ or _checkpoint_1000000
        patterns = [
            r'_(\d{6,})_',  # 6+ digits surrounded by underscores
            r'checkpoint[_-](\d+)',  # checkpoint_123456
            r'step[_-](\d+)',  # step_123456
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_timestamp(self, filename: str) -> Optional[str]:
        """Extract timestamp from filename."""
        # Look for patterns like 20231201 or 2023-12-01
        patterns = [
            r'(\d{8})',  # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})',  # YYYY_MM_DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None
    
    def load_agent(self, agent_id: str, **kwargs) -> SB3Agent:
        """
        Load a specific agent by ID (filename or path).
        
        Args:
            agent_id: Agent identifier (filename or path)
            **kwargs: Additional arguments for SB3Agent
            
        Returns:
            Loaded SB3Agent
        """
        # Check if already loaded
        if agent_id in self.loaded_agents:
            return self.loaded_agents[agent_id]
        
        # Find agent in registry
        agent_info = None
        for info in self.agent_registry:
            if agent_id in [info['filename'], info['path'], Path(info['path']).stem]:
                agent_info = info
                break
        
        if agent_info is None:
            # Try direct path
            agent_path = Path(agent_id)
            if agent_path.exists():
                agent = SB3Agent(agent_path=str(agent_path), **kwargs)
            else:
                raise FileNotFoundError(f"Agent not found: {agent_id}")
        else:
            agent = SB3Agent(
                agent_path=agent_info['path'],
                algorithm=agent_info.get('algorithm'),
                **kwargs
            )
        
        # Cache the loaded agent
        self.loaded_agents[agent_id] = agent
        return agent
    
    def load_agents_by_criteria(self, **criteria) -> List[Tuple[str, SB3Agent]]:
        """
        Load multiple agents based on criteria.
        
        Args:
            **criteria: Filtering criteria (algorithm, environment, etc.)
            
        Returns:
            List of (agent_id, SB3Agent) tuples
        """
        matching_agents = []
        
        for agent_info in self.agent_registry:
            # Check if agent matches criteria
            matches = True
            
            for key, value in criteria.items():
                if key in agent_info:
                    if isinstance(value, list):
                        if agent_info[key] not in value:
                            matches = False
                            break
                    else:
                        if agent_info[key] != value:
                            matches = False
                            break
            
            if matches:
                try:
                    agent_id = agent_info['filename']
                    agent = self.load_agent(agent_id)
                    matching_agents.append((agent_id, agent))
                except Exception as e:
                    print(f"Error loading agent {agent_info['filename']}: {e}")
        
        return matching_agents
    
    def get_checkpoint_sequence(self, base_name: str) -> List[Dict[str, Any]]:
        """
        Get sequence of checkpoints for a specific agent training run.
        
        Args:
            base_name: Base name pattern to match
            
        Returns:
            List of agent info sorted by checkpoint
        """
        matching_agents = []
        
        for agent_info in self.agent_registry:
            if base_name.lower() in agent_info['filename'].lower():
                if agent_info.get('checkpoint') is not None:
                    matching_agents.append(agent_info)
        
        # Sort by checkpoint
        matching_agents.sort(key=lambda x: x.get('checkpoint', 0))
        return matching_agents
    
    def load_checkpoint_sequence(self, base_name: str) -> List[Tuple[int, SB3Agent]]:
        """
        Load a sequence of checkpoints as agents.
        
        Args:
            base_name: Base name pattern to match
            
        Returns:
            List of (checkpoint, SB3Agent) tuples sorted by checkpoint
        """
        checkpoint_info = self.get_checkpoint_sequence(base_name)
        loaded_checkpoints = []
        
        for info in checkpoint_info:
            try:
                agent = self.load_agent(info['filename'])
                checkpoint = info.get('checkpoint', 0)
                loaded_checkpoints.append((checkpoint, agent))
            except Exception as e:
                print(f"Error loading checkpoint {info['filename']}: {e}")
        
        return loaded_checkpoints
    
    def get_agent_registry(self) -> List[Dict[str, Any]]:
        """Get the full agent registry."""
        return self.agent_registry.copy()
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the agent registry."""
        if not self.agent_registry:
            return {
                'total_agents': 0,
                'algorithms': {},
                'environments': {},
                'checkpoint_range': {'min': None, 'max': None, 'count': 0},
                'total_size_mb': 0.0,
                'agents_with_metadata': 0
            }
        
        algorithms = [info.get('algorithm') for info in self.agent_registry if info.get('algorithm')]
        environments = [info.get('environment') for info in self.agent_registry if info.get('environment')]
        checkpoints = [info.get('checkpoint') for info in self.agent_registry if info.get('checkpoint')]
        
        summary = {
            'total_agents': len(self.agent_registry),
            'algorithms': {alg: algorithms.count(alg) for alg in set(algorithms)} if algorithms else {},
            'environments': {env: environments.count(env) for env in set(environments)} if environments else {},
            'checkpoint_range': {
                'min': min(checkpoints) if checkpoints else None,
                'max': max(checkpoints) if checkpoints else None,
                'count': len(checkpoints)
            },
            'total_size_mb': sum(info.get('file_size', 0) for info in self.agent_registry) / (1024 * 1024) if self.agent_registry else 0,
            'agents_with_metadata': sum(1 for info in self.agent_registry if 'metadata' in info)
        }
        
        return summary
    
    def find_best_agents(self, metric: str = 'mean_reward', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the best performing agents based on a metric.
        
        Args:
            metric: Metric to use for ranking
            top_k: Number of top agents to return
            
        Returns:
            List of top agent info dictionaries
        """
        agents_with_metrics = []
        
        for agent_info in self.agent_registry:
            metadata = agent_info.get('metadata', {})
            performance = metadata.get('performance_metrics', {})
            
            if metric in performance:
                agent_info_copy = agent_info.copy()
                agent_info_copy['metric_value'] = performance[metric]
                agents_with_metrics.append(agent_info_copy)
        
        # Sort by metric (descending)
        agents_with_metrics.sort(key=lambda x: x['metric_value'], reverse=True)
        
        return agents_with_metrics[:top_k]
    
    def create_agent_comparison_dataset(
        self, 
        agent_ids: List[str],
        episodes_per_agent: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create a comparison dataset from multiple agents.
        
        Args:
            agent_ids: List of agent identifiers
            episodes_per_agent: Number of episodes per agent
            
        Returns:
            Dictionary mapping agent IDs to episode data
        """
        comparison_data = {}
        
        for agent_id in agent_ids:
            try:
                agent = self.load_agent(agent_id)
                episodes = agent.run_multiple_episodes(
                    num_episodes=episodes_per_agent,
                    record_frames=True
                )
                comparison_data[agent_id] = episodes
                print(f"Generated {len(episodes)} episodes for agent {agent_id}")
                
            except Exception as e:
                print(f"Error generating data for agent {agent_id}: {e}")
                comparison_data[agent_id] = []
        
        return comparison_data 