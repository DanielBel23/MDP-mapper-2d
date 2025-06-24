from dataclasses import dataclass

# ---------- state is where the agent is located ----------
@dataclass(frozen=True)
class State:
    __slots__ = ('y', 'x')
    y: int
    x: int

# ---------- MDP for grid map ----------
class MDPComponents:
    MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))
    BARRIER, ACTIVE, TERMINAL = 0, 1, -1

    def __init__(self, states_arr, rewards_arr, transition_probabilities, discount_factor):
        self.states_arr = states_arr
        self.rewards_arr = rewards_arr
        self.p = transition_probabilities
        self.gamma = discount_factor
        self.state_space = self._build_state_space()

    # ---------- MDP components ----------

    def states(self):
        return self.state_space.values()

    def is_terminal(self, s: State):
        if self.states_arr[s.y][s.x] == self.TERMINAL:
            return True
        return False

    def actions(self, s: State):
        possible_actions = []
        if self.is_terminal(s):
            return []
        for move in self.MOVES:
            if self._can_move(move, s):
                possible_actions.append(move)
        return possible_actions

    def transition_probabilities(self, s: State, action):
        added_prob = 0
        p_fail = (1-self.p)/2
        diag1, diag2 = self._orthogonals(action)
        possible_transitions = []
        if self._can_move(diag1, s):
            possible_transitions.append([self._apply_action(s, diag1), p_fail])
        else:
            added_prob += p_fail
        if self._can_move(diag2, s):
            possible_transitions.append([self._apply_action(s, diag2), p_fail])
        else:
            added_prob += p_fail
        possible_transitions.append([self._apply_action(s, action), added_prob+self.p])
        return possible_transitions

    def reward(self, next_state: State):
        return self.rewards_arr[next_state.y][next_state.x]

    def discount_factor(self):
        return self.gamma

    # ---------- supporting actions ----------
    def _apply_action(self, s: State, action):
        return  self.state_space.get((s.y+action[0], s.x+action[1]))

    def _can_move(self, vector, given_state):
        new_y, new_x = given_state.y+vector[0], given_state.x+vector[1]
        if new_x<0 or new_x>self.states_arr.shape[1]-1:
            return False
        if new_y<0 or new_y>self.states_arr.shape[0]-1:
            return False
        if self.states_arr[new_y][new_x] != 0:
            return True
        return False

    def _build_state_space(self):
        states = {}
        for y in range(self.states_arr.shape[0]):
            for x in range(self.states_arr.shape[1]):
                if self.states_arr[y][x] != self.BARRIER:
                    states[(y, x)] = State(y, x)
        return states

    def _orthogonals(self, action):
        if action[0] == 0:
            return (1, action[1]), (-1, action[1])
        else:
            return (action[0], 1), (action[0], -1)