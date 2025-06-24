import sys
import argparse
from inputHandler import load_and_validate
import ValueIteration
import PolicyIteration
import visualization as viz

def main():
    parser = argparse.ArgumentParser(
        description="MDP runner: load data and apply algorithms.")
    parser.add_argument('npz_path',
                        help='Path to the .npz input file')
    parser.add_argument('algorithm',
                        choices=['ValueIteration', 'PolicyIteration'],
                        help='Name of the algorithm to run')
    parser.add_argument('--keep-values',
                        action='store_true',
                        help=('For PolicyIteration only: reuse previous value '
                              'estimates (warm start). Omit to reset each iteration.'))
    parser.add_argument('--graph',
                        action='store_true',
                        help=('When running PolicyIteration, also generate a '
                              'graph that shows how many simplified value-'
                              'iteration sweeps were required in each '
                              'policy-iteration step.'))
    args = parser.parse_args()

    valid, matrices = load_and_validate(args.npz_path)
    if not valid:
        sys.exit(1)
    states_arr, rewards_arr = matrices

    # algorithm-wide defaults
    p       = 0.8
    gamma   = 0.9
    epsilon = 1e-3
    fname   = args.npz_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]

    if args.algorithm == 'ValueIteration':
        values_history, policy = ValueIteration.value_iteration(
            states_arr, rewards_arr, p, gamma, epsilon)
        viz.animate_values(values_history, fname, name='ValueIteration')
        viz.plot_policy(policy, fname, name='ValueIteration')


    elif args.algorithm == 'PolicyIteration':
        reset_values = not args.keep_values
        values_history, policy, sweeps = PolicyIteration.policy_iteration(
            states_arr, rewards_arr,
            p, gamma, epsilon,
            reset_values=reset_values)
        outname = f"{fname}_PolicyIteration"
        viz.animate_values(values_history, outname, name='PolicyIteration')
        viz.plot_policy(policy, outname, name='PolicyIteration')
        if args.graph:
            viz.plot_evaluation_sweeps(sweeps, outname, name='PolicyIteration')

    else:
        print(f"Algorithm '{args.algorithm}' not supported.")
        sys.exit(1)


if __name__ == '__main__':
    main()
