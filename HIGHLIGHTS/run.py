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
    parser.add_argument("--seed", help="Random generator seed for environment", type=int, default=0)
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
    
    # Enhanced seeding options for fair comparison between methods
    parser.add_argument('--trace-seeds', help='specific seeds for each trace (comma-separated)', type=str, default=None)
    parser.add_argument('--seed-mode', help='seeding strategy for fair method comparison', 
                       choices=['fixed', 'sequential', 'random', 'trace-specific'], default='fixed')
    parser.add_argument('--base-seed', help='base seed for sequential seeding mode', type=int, default=None)
    parser.add_argument('--deterministic-eval', help='use deterministic policy evaluation', action='store_true', default=True)
    parser.add_argument('--save-seeds', help='save used seeds to metadata for reproducibility', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Process seeding arguments
    if args.base_seed is None:
        args.base_seed = args.seed
    
    # Parse trace-specific seeds if provided
    if args.trace_seeds:
        try:
            trace_seeds = [int(s.strip()) for s in args.trace_seeds.split(',')]
            if len(trace_seeds) != args.n_traces:
                print(f"Warning: {len(trace_seeds)} trace seeds provided for {args.n_traces} traces.")
                print(f"Will cycle through provided seeds or use default seeding for missing seeds.")
            args.parsed_trace_seeds = trace_seeds
        except ValueError:
            print("Error: Invalid trace seeds format. Use comma-separated integers.")
            exit(1)
    else:
        args.parsed_trace_seeds = None
    
    # Generate seeds based on mode
    if args.seed_mode == 'fixed':
        # All traces use the same environment seed (original behavior)
        args.trace_seed_list = [args.seed] * args.n_traces
    elif args.seed_mode == 'sequential':
        # Sequential seeds starting from base_seed
        args.trace_seed_list = [args.base_seed + i for i in range(args.n_traces)]
    elif args.seed_mode == 'random':
        # Random seeds (but reproducible if base_seed is set)
        import random
        random.seed(args.base_seed)
        args.trace_seed_list = [random.randint(0, 999999) for _ in range(args.n_traces)]
    elif args.seed_mode == 'trace-specific':
        # Use provided trace seeds or fall back to sequential
        if args.parsed_trace_seeds:
            # Extend or cycle through provided seeds if needed
            args.trace_seed_list = []
            for i in range(args.n_traces):
                seed_idx = i % len(args.parsed_trace_seeds)
                args.trace_seed_list.append(args.parsed_trace_seeds[seed_idx])
        else:
            print("Error: trace-specific mode requires --trace-seeds")
            exit(1)
    
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
    print(f"Deterministic evaluation: {args.deterministic_eval}")
    print("="*60)
    print("Seeding Configuration:")
    print(f"Base seed: {args.base_seed}")
    print(f"Seed mode: {args.seed_mode}")
    if args.seed_mode == 'trace-specific' and args.parsed_trace_seeds:
        print(f"Trace seeds: {args.parsed_trace_seeds}")
    print(f"Generated trace seeds: {args.trace_seed_list[:5]}{'...' if len(args.trace_seed_list) > 5 else ''}")
    print("="*60)
    
    # Run HIGHLIGHTS
    try:
        main(args)
        print("\nHIGHLIGHTS completed successfully!")
        if args.output_dir:
            print(f"Results saved to: {args.output_dir}")
        print("\nFor fair method comparison, use the same seed configuration:")
        print(f"  --seed-mode {args.seed_mode} --base-seed {args.base_seed}")
        if args.seed_mode == 'trace-specific':
            print(f"  --trace-seeds {','.join(map(str, args.parsed_trace_seeds))}")
    except Exception as e:
        print(f"\nError running HIGHLIGHTS: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
