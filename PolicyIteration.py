# PolicyIteration.py
import numpy as np
from typing import List, Tuple, Dict
from MDPComponents import MDPComponents, State

# Helper mappings
_ACTION_TO_CHAR: Dict[Tuple[int, int], str] = {
    (-1, 0): '^',  # up
    (1, 0):  'v',  # down
    (0, 1):  '>',  # right
    (0, -1): '<',  # left
}
_CHAR_TO_ACTION: Dict[str, Tuple[int, int]] = {v: k for k, v in _ACTION_TO_CHAR.items()}

def action_to_char(action: Tuple[int, int]) -> str:
    return _ACTION_TO_CHAR.get(action, '+')

def char_to_action(ch: str) -> Tuple[int, int]:
    return _CHAR_TO_ACTION.get(ch)

def simplified_value_iteration(states_arr: np.ndarray,
                               rewards_arr: np.ndarray,
                               p: float,
                               gamma: float,
                               epsilon: float = 1e-3):
    from ValueIteration import value_iteration
    history, policy = value_iteration(states_arr, rewards_arr, p, gamma, epsilon)
    return history[-1], policy

def default_policy(states_arr: np.ndarray, mdp: MDPComponents) -> np.ndarray:
    rows, cols = states_arr.shape
    policy = np.full((rows, cols), '', dtype='<U1')
    preferred_moves = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # ^,v,>,<

    for y in range(rows):
        for x in range(cols):
            cell = states_arr[y, x]
            if cell == mdp.BARRIER:
                policy[y, x] = 'x'
            elif cell == mdp.TERMINAL:
                policy[y, x] = 'o'
            else:
                s = mdp.state_space[(y, x)]
                legals = mdp.actions(s)
                pick = next((m for m in preferred_moves if m in legals), None)
                policy[y, x] = action_to_char(pick) if pick else '+'
    return policy

def q_value(mdp: MDPComponents, s: State, a: Tuple[int, int],
            V: np.ndarray, gamma: float) -> float:
    total = 0.0
    for ns, prob in mdp.transition_probabilities(s, a):
        total += prob * (mdp.reward(ns) + gamma * V[ns.y, ns.x])
    return total

def policy_evaluation(mdp: MDPComponents,
                      policy: np.ndarray,
                      gamma: float,
                      epsilon: float = 1e-3,
                      V_init: np.ndarray = None) -> Tuple[np.ndarray, int]:
    V = np.zeros_like(mdp.states_arr, dtype=float) if V_init is None else V_init.copy()
    sweeps = 0

    while True:
        sweeps += 1
        delta = 0.0
        V_old = V.copy()
        for s in mdp.states():
            y, x = s.y, s.x
            if mdp.is_terminal(s):
                V[y, x] = mdp.reward(s)
                continue

            sym = policy[y, x]
            if sym in ('x', 'o'):
                continue

            if sym == '+':
                actions = mdp.actions(s)
            else:
                a = char_to_action(str(sym))
                actions = [a] if a else []

            if not actions:
                V[y, x] = 0
                continue

            expected = sum(q_value(mdp, s, a, V_old, gamma) for a in actions)
            V[y, x] = expected / len(actions)
            delta = max(delta, abs(V[y, x] - V_old[y, x]).item())
        if delta < epsilon:
            break

    return V, sweeps

def policy_iteration(states_arr: np.ndarray,
                     rewards_arr: np.ndarray,
                     p: float,
                     gamma: float,
                     epsilon: float = 1e-3,
                     reset_values: bool = True
                     ) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    mdp = MDPComponents(states_arr, rewards_arr, p, gamma)
    policy = default_policy(states_arr, mdp)
    V = np.zeros_like(states_arr, dtype=float)
    values_history: List[np.ndarray] = []
    sweeps_history: List[int] = []

    while True:
        V, sweeps = policy_evaluation(mdp, policy, gamma, epsilon,
                                      None if reset_values else V)
        values_history.append(V.copy())
        sweeps_history.append(sweeps)

        unchanged = True
        new_pol = policy.copy()
        for s in mdp.states():
            if mdp.is_terminal(s):
                continue
            acts = mdp.actions(s)
            if not acts:
                continue

            best_a = max(acts, key=lambda a: q_value(mdp, s, a, V, gamma))
            best_q = q_value(mdp, s, best_a, V, gamma)

            sym = policy[s.y, s.x]
            if sym == '+':
                curr_q = np.mean([q_value(mdp, s, a, V, gamma) for a in acts])
            else:
                curr_q = q_value(mdp, s, char_to_action(str(sym)), V, gamma)

            if best_q > curr_q + 1e-12:
                new_pol[s.y, s.x] = action_to_char(best_a)
                unchanged = False

        policy = new_pol
        if unchanged:
            break
        if reset_values:
            V = np.zeros_like(states_arr, dtype=float)

    return values_history, policy, sweeps_history

# Note: standalone CLI removed—use MDP.py as the sole entry point.