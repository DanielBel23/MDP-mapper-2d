# MDP Solver: Value Iteration & Policy Iteration

This project implements a Markov Decision Process (MDP) solver using **Value Iteration** and **Policy Iteration**, visualized step-by-step with Python.

---

## Project Structure

├── MDP.py                # Main runner script  
├── inputHandler.py       # Loads and validates .npz input files  
├── MDPComponents.py      # Core MDP logic (states, actions, transitions, etc.)  
├── ValueIteration.py     # Value Iteration algorithm  
├── PolicyIteration.py    # Policy Iteration algorithm  
├── visualization.py      # Visualization (plots and animations)  
├── Output_jpgs/          # Output folder for generated images  
└── README.md             # This file  

---

## How to Run

### Requirements

- Python 3.x
- `numpy`
- `matplotlib`

Install required packages:

```bash
pip install numpy matplotlib
```

---

### Input Format

Provide a `.npz` file containing:

- A **state matrix**: integers with values `-1`, `0`, or `1`
  - `-1` = terminal state
  - `0`  = barrier/wall
  - `1`  = regular cell
- A **reward matrix**: same shape, with numeric values

---

### Command Line Usage

```bash
python MDP.py <file.npz> <algorithm> [--keep-values] [--graph]
```

#### Arguments

- `ValueIteration` — run Value Iteration algorithm.
- `PolicyIteration` — run Policy Iteration algorithm.
  - `--keep-values` — reuse previous value estimates across iterations.
  - `--graph` — generate a graph showing the number of simplified value iterations per policy update.

---

### Examples

```bash
# Value Iteration
python MDP.py input.npz ValueIteration

# Policy Iteration (reset values each time)
python MDP.py input.npz PolicyIteration

# Policy Iteration (reuse values, with graph)
python MDP.py input.npz PolicyIteration --keep-values --graph
```

---

## Output

Visuals are saved in the `Output_jpgs/` folder:

- Value iteration animations
- Final policy diagram
- (Optional) Graph of simplified evaluation sweeps per iteration

---

## MDP Logic Overview (from `MDPComponents.py`)

- `states()` — returns all usable states.
- `is_terminal(state)` — checks if state is terminal.
- `actions(state)` — returns legal actions for a state.
- `transition_probabilities(state, action)` — gives next states and probabilities.
- `reward(state)` — returns reward for a state.
- `discount_factor()` — returns γ (gamma), the discount factor.

---

## Notes

- Convergence threshold ε = **0.001**
- Lower discount factors (γ) cause short-sighted behavior
- Higher transition probabilities (p) improve convergence
- Scaling rewards affects convergence time, not policy outcome

---

## Author

**Daniel Belizki**
