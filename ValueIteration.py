import numpy as np
from MDPComponents import MDPComponents

def value_iteration(states_arr: np.ndarray,
                    rewards_arr: np.ndarray,
                    p: float,
                    gamma: float,
                    epsilon: float = 1e-3):
    """
    Perform value iteration and return the history of value matrices and the optimal policy.
    """
    mdp = MDPComponents(states_arr, rewards_arr, p, gamma)
    values_history = []

    # Initialize V to zeros
    V = np.zeros_like(states_arr, dtype=float)

    while True:
        delta = 0
        V_old = V.copy()

        # Bellman update for each non-barrier state
        for state in mdp.states():
            y, x = state.y, state.x
            if not mdp.actions(state):
                V[y, x] = mdp.reward(state)
                continue
            if mdp.is_terminal(state):
                V[y, x] = mdp.reward(state)
            else:
                action_values = []
                for action in mdp.actions(state):
                    total = 0
                    for next_state, prob in mdp.transition_probabilities(state, action):
                        total += prob * (mdp.reward(next_state) + gamma * V_old[next_state.y, next_state.x])
                    action_values.append(total)
                best = max(action_values)
                V[y, x] = best
                delta = max(delta, abs(best - V_old[y, x]))

        values_history.append(V.copy())

        if delta < epsilon:
            break

    # Extract policy
    policy = np.full(states_arr.shape, '', dtype='<U1')
    for y in range(states_arr.shape[0]):
        for x in range(states_arr.shape[1]):
            cell = states_arr[y, x]
            if cell == 0:
                policy[y, x] = 'x'
            elif cell == -1:
                policy[y, x] = 'o'
            else:
                state = mdp.state_space[(y, x)]
                if not mdp.actions(state):
                    policy[y, x] = 'o'
                    continue
                q_values = {}
                for action in mdp.actions(state):
                    total = 0
                    for next_state, prob in mdp.transition_probabilities(state, action):
                        total += prob * (mdp.reward(next_state) + gamma * V[next_state.y, next_state.x])
                    q_values[action] = total

                max_q = max(q_values.values())
                best_actions = [a for a, q in q_values.items() if abs(q - max_q) < 1e-8]
                if len(best_actions) > 1:
                    policy[y, x] = '+'
                else:
                    dy, dx = best_actions[0]
                    if (dy, dx) == (-1, 0): policy[y, x] = '^'
                    elif (dy, dx) == (1, 0):  policy[y, x] = 'v'
                    elif (dy, dx) == (0, 1):  policy[y, x] = '>'
                    elif (dy, dx) == (0, -1): policy[y, x] = '<'

    return values_history, policy