import random

import gym as gym
import pandas as pd
from os.path import join, abspath

from highlights.get_agent import get_agent
from highlights.get_traces import get_traces
from highlights.utils import pickle_save, pickle_load
from highlights.highlights_state_selection import compute_states_importance, highlights, \
    highlights_div
from highlights.Importance_by_trajectory import states_to_trajectories, trajectories_by_importance


def get_highlights(args):
    """Load traces and state dictionary"""
    if args.load_dir:
        """Load traces and state dictionary"""
        traces = pickle_load(join(args.load_dir, 'Traces.pkl'))
        states = pickle_load(join(args.load_dir, 'States.pkl'))
        if args.verbose: print(f"Highlights {15 * '-' + '>'} Traces & States Loaded")
    else:
        env, agent = get_agent(args)
        env.args = args
        traces, states = get_traces(env, agent, args)
        env.close()
        # Handle different gym versions for registry cleanup
        try:
            if hasattr(gym.envs.registration.registry, 'env_specs'):
                del gym.envs.registration.registry.env_specs[env.spec.id]
            elif hasattr(gym.envs.registration.registry, 'all'):
                # Newer gym versions
                if env.spec.id in gym.envs.registration.registry.all():
                    del gym.envs.registration.registry.all()[env.spec.id]
        except (AttributeError, KeyError):
            # Skip cleanup if registry structure is different
            pass
        pickle_save(traces, join(abspath(args.output_dir), 'Traces.pkl'))
        pickle_save(states, join(abspath(args.output_dir), 'States.pkl'))

    """Save data used for this run in output dir"""
    pickle_save(traces, join(args.output_dir, 'Traces.pkl'))
    pickle_save(states, join(args.output_dir, 'States.pkl'))

    """highlights algorithm"""
    a, b, c = states[(0, 0)].image.shape
    reshape = a * b * c
    data = {'state': list(states.keys()),
            'q_values': [x.observed_actions for x in states.values()],
            'features': [x.image.reshape(reshape) for x in states.values()]}
    q_values_df = pd.DataFrame(data)

    """importance by state"""
    q_values_df = compute_states_importance(q_values_df, compare_to=args.state_importance)
    highlights_df = q_values_df
    state_importance_dict = dict(zip(highlights_df["state"], highlights_df["importance"]))

    """get highlights"""
    if args.importance_type == "single_state":
        """highlights importance by single state importance"""
        trace_lengths = {k: len(v.states) for k, v in enumerate(traces)}
        num_highlights = getattr(args, 'num_highlights', getattr(args, 'num_trajectories', 5))
        if args.highlights_div:
            """threshold for divergence based on middle state of first trace"""
            i = trace_lengths[0] // 2
            threshold = args.div_coefficient * (
                sum(states[(0, i)].image.reshape(reshape) - states[(0, i + 1)].image.reshape(reshape)))
            """get highlights"""
            summary_states = highlights_div(highlights_df, trace_lengths, num_highlights,
                                            args.trajectory_length, args.minimum_gap,
                                            threshold=threshold)
        else:
            summary_states = highlights(highlights_df, trace_lengths, num_highlights,
                                        args.trajectory_length, args.minimum_gap)
        all_trajectories = states_to_trajectories(summary_states, state_importance_dict)
        summary_trajectories = all_trajectories

    else:
        """highlights importance by trajectory"""
        all_trajectories, summary_trajectories = \
            trajectories_by_importance(traces, state_importance_dict, args)

    # random order
    if args.randomized: random.shuffle(summary_trajectories)
    """Save trajectories used for this run in output dir"""
    pickle_save(all_trajectories, join(args.output_dir, 'Trajectories.pkl'))

    return states, summary_trajectories
