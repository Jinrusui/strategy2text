"""
Trace collection for stable-baselines3 models
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from highlights.utils import Trace, State
from highlights.value_extractor import extract_q_values, compute_state_importance


def get_traces(environment, model, args):
    """
    Obtain traces and state dictionary from stable-baselines3 model
    
    Args:
        environment: Environment (can be VecEnv or regular gym env)
        model: Trained stable-baselines3 model
        args: Arguments containing trace collection parameters
    
    Returns:
        tuple: (execution_traces, states_dictionary)
    """
    execution_traces, states_dictionary = [], {}
    
    # Handle VecEnv vs regular env
    is_vec_env = isinstance(environment, VecEnv)
    
    # Get trace seeds from args
    trace_seeds = getattr(args, 'trace_seed_list', [args.seed] * args.n_traces)
    deterministic_eval = getattr(args, 'deterministic_eval', True)
    
    if args.verbose:
        print(f"Using trace seeds: {trace_seeds}")
        print(f"Deterministic evaluation: {deterministic_eval}")
    
    for i in range(args.n_traces):
        # Use specific seed for this trace
        trace_seed = trace_seeds[i] if i < len(trace_seeds) else args.seed
        
        get_single_trace(environment, model, i, execution_traces, states_dictionary, 
                        args, is_vec_env, trace_seed=trace_seed, 
                        deterministic=deterministic_eval)
        if args.verbose: 
            print(f"\tTrace {i} (seed: {trace_seed}) {15*'-'+'>'} Obtained")
    
    if args.verbose: 
        print(f"Highlights {15*'-'+'>'} Traces & States Generated")
    
    return execution_traces, states_dictionary


def get_single_trace(env, model, trace_idx, agent_traces, states_dict, args, is_vec_env=False, 
                    trace_seed=None, deterministic=True):
    """
    Implement a single trace while using the Trace and State classes
    
    Args:
        env: Environment
        model: Stable-baselines3 model
        trace_idx: Index of current trace
        agent_traces: List to append trace to
        states_dict: Dictionary to store states
        args: Arguments
        is_vec_env: Whether environment is vectorized
        trace_seed: Specific seed for this trace (for reproducible scenarios)
        deterministic: Whether to use deterministic policy evaluation
    """
    import numpy as np
    from stable_baselines3.common.utils import set_random_seed
    
    # Set seed for this specific trace
    if trace_seed is not None:
        set_random_seed(trace_seed)
        if is_vec_env:
            env.seed(trace_seed)
        else:
            env.seed(trace_seed)
            # Also seed the action space for stochastic environments
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
                env.action_space.seed(trace_seed)
    
    trace = Trace()
    
    # Reset environment
    if is_vec_env:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
    else:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
    
    done = False
    step_count = 0
    max_steps = getattr(args, 'max_steps_per_trace', 1000)  # Prevent infinite loops
    
    while not done and step_count < max_steps:
        # Get action from model (use deterministic parameter)
        if is_vec_env:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action, _ = model.predict(obs.reshape(1, -1) if obs.ndim > 0 else obs, deterministic=deterministic)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()
        
        # Extract Q-values for current observation
        try:
            if is_vec_env:
                # For vectorized environments, extract from first environment
                current_obs = obs[0] if isinstance(obs, np.ndarray) and len(obs.shape) > len(model.observation_space.shape) else obs
                q_values = extract_q_values(model, current_obs)
            else:
                q_values = extract_q_values(model, obs)
        except Exception as e:
            print(f"Warning: Could not extract Q-values: {e}")
            # Get number of actions from environment or model
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                n_actions = env.action_space.n
            elif hasattr(model, 'action_space') and hasattr(model.action_space, 'n'):
                n_actions = model.action_space.n
            else:
                n_actions = 4  # Default for Atari games
            q_values = np.zeros(n_actions)
        
        # Take action in environment
        if is_vec_env:
            obs_next, reward, done, info = env.step(action)
            if isinstance(obs_next, tuple):
                obs_next = obs_next[0]
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(info, list):
                info = info[0] if info else {}
        else:
            step_result = env.step(action)
            if len(step_result) == 4:
                obs_next, reward, done, info = step_result
            else:  # Handle new gym API with 5 return values
                obs_next, reward, terminated, truncated, info = step_result
                done = terminated or truncated
        
        # Generate state image
        try:
            if is_vec_env:
                state_img = env.render()
                if isinstance(state_img, list):
                    state_img = state_img[0]
            else:
                state_img = env.render()
        except Exception as e:
            print(f"Warning: Could not render state: {e}")
            # Create dummy image
            state_img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Compute features (using flattened image for now)
        try:
            if state_img is not None:
                features = state_img.flatten()
            else:
                features = np.zeros(64*64*3)
        except:
            features = np.zeros(64*64*3)
        
        # Create state ID
        state_id = (trace_idx, trace.length)
        
        # Create State object
        state = State(
            name=state_id,
            obs=obs.copy() if isinstance(obs, np.ndarray) else obs,
            action_vector=q_values,
            feature_vector=features,
            img=state_img
        )
        
        # Store state
        states_dict[state_id] = state
        
        # Update trace
        trace.update(obs, reward, done, info, action, state_id)
        
        # Move to next observation
        obs = obs_next
        step_count += 1
        
        # Handle episode termination
        if done:
            # Optionally collect final state
            if hasattr(args, 'collect_final_state') and args.collect_final_state:
                try:
                    final_q_values = extract_q_values(model, obs)
                    final_state_img = env.render() if not is_vec_env else env.render()[0]
                    final_features = final_state_img.flatten() if final_state_img is not None else np.zeros(64*64*3)
                    final_state_id = (trace_idx, trace.length)
                    final_state = State(
                        name=final_state_id,
                        obs=obs.copy() if isinstance(obs, np.ndarray) else obs,
                        action_vector=final_q_values,
                        feature_vector=final_features,
                        img=final_state_img
                    )
                    states_dict[final_state_id] = final_state
                except:
                    pass  # Skip final state if there's an error
    
    # Store final game score if available
    if hasattr(env, 'get_episode_rewards'):
        try:
            episode_rewards = env.get_episode_rewards()
            if episode_rewards:
                trace.game_score = episode_rewards[-1]
        except:
            pass
    
    # Store trace seed in trace object for reference
    trace.trace_seed = trace_seed
    
    agent_traces.append(trace)


def validate_trace_collection(traces, states_dict, args):
    """
    Validate collected traces and states
    
    Args:
        traces: List of traces
        states_dict: Dictionary of states
        args: Arguments
    
    Returns:
        bool: True if validation passes
    """
    if not traces:
        print("Error: No traces collected")
        return False
    
    if not states_dict:
        print("Error: No states collected")
        return False
    
    # Check trace lengths
    for i, trace in enumerate(traces):
        if trace.length == 0:
            print(f"Warning: Trace {i} has zero length")
        elif trace.length > getattr(args, 'max_steps_per_trace', 1000):
            print(f"Warning: Trace {i} is very long ({trace.length} steps)")
    
    # Check state Q-values
    invalid_q_values = 0
    for state_id, state in states_dict.items():
        if state.observed_actions is None or len(state.observed_actions) == 0:
            invalid_q_values += 1
    
    if invalid_q_values > 0:
        print(f"Warning: {invalid_q_values} states have invalid Q-values")
    
    print(f"Validation: {len(traces)} traces, {len(states_dict)} states collected")
    return True
