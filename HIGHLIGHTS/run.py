"""
HIGHLIGHTS for rl-baselines3-zoo: Agent Policy Summarization for Value-Based RL
"""

import argparse
from os.path import abspath
from huggingface_sb3 import EnvironmentName

from highlights.main import main
from rl_zoo3.utils import ALGOS, StoreDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIGHLIGHTS for rl-baselines3-zoo')
    
    # rl-baselines3-zoo compatible arguments
    parser.add_argument("--env", help="Environment ID", type=EnvironmentName, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="../rl-baselines3-zoo/rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="dqn", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--load-best", action="store_true", default=False, help="Load best model instead of last model if available")
    parser.add_argument("--load-checkpoint", type=int, help="Load checkpoint instead of last model if available")
    parser.add_argument("--load-last-checkpoint", action="store_true", default=False, help="Load last checkpoint instead of last model if available")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues")
    parser.add_argument("--n-envs", help="Number of environments", default=1, type=int)
    
    # HIGHLIGHTS specific arguments
    parser.add_argument('--load_dir', help='path to existing traces', type=str, default=None)
    parser.add_argument('-fps', '--fps', help='summary video fps', type=int, default=5)
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain', type=int, default=10)
    parser.add_argument('-k', '--num_highlights', help='number of highlights trajectories to obtain', type=int, default=5)
    parser.add_argument('-l', '--trajectory_length', help='length of highlights trajectories', type=int, default=10)
    parser.add_argument('-v', '--verbose', help='print information to the console', action='store_true', default=True)
    parser.add_argument('-overlapLim', '--overlay_limit', help='# overlapping', type=int, default=3)
    parser.add_argument('-minGap', '--minimum_gap', help='minimum gap between trajectories', type=int, default=0)
    parser.add_argument('-rand', '--randomized', help='randomize order of summary trajectories', type=bool, default=True)
    parser.add_argument('-impMeth', '--importance_type', help='importance by state or trajectory', default='single_state')
    parser.add_argument('-impState', '--state_importance', help='method calculating state importance', default='second')
    parser.add_argument('--highlights_div', help='use diversity measures', type=bool, default=False)
    parser.add_argument('--div_coefficient', help='diversity coefficient', type=int, default=2)
    parser.add_argument('--pause', help='pause frames at start/end of videos', type=int, default=0)
    parser.add_argument('--max_steps_per_trace', help='maximum steps per trace to prevent infinite loops', type=int, default=1000)
    parser.add_argument('--collect_final_state', help='collect final state of episode', action='store_true', default=False)
    parser.add_argument('--output_dir', help='output directory for results', type=str, default=None)
    
    args = parser.parse_args()
    
    # Validate algorithm is value-based
    value_based_algos = ['dqn', 'qrdqn', 'ddqn']
    if args.algo.lower() not in value_based_algos:
        print(f"Warning: Algorithm '{args.algo}' may not be value-based.")
        print(f"HIGHLIGHTS works best with value-based algorithms: {value_based_algos}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            exit(0)
    
    # Print configuration
    print("="*60)
    print("HIGHLIGHTS Configuration:")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")
    print(f"Model folder: {args.folder}")
    print(f"Experiment ID: {args.exp_id}")
    print(f"Number of traces: {args.n_traces}")
    print(f"Number of highlights: {args.num_highlights}")
    print(f"Trajectory length: {args.trajectory_length}")
    print(f"State importance method: {args.state_importance}")
    print(f"Use diversity: {args.highlights_div}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Run HIGHLIGHTS
    try:
        main(args)
        print("\nHIGHLIGHTS completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"\nError running HIGHLIGHTS: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
