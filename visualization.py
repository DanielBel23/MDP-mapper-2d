import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the output directory exists
os.makedirs('Output_jpgs', exist_ok=True)

AUTHOR = "Daniel Belizki"

def animate_values(values_list: list, fname: str, name: str = AUTHOR, interval: float = 0.5):
    max_abs = max(np.max(np.abs(mat)) for mat in values_list)
    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(values_list[0], vmin=-max_abs, vmax=max_abs, cmap='seismic')
    title = ax.set_title(f'Value Iteration: {name} (Iteration 0)')
    cbar = fig.colorbar(im)

    for i, mat in enumerate(values_list):
        im.set_data(mat)
        title.set_text(f'Value Iteration: {name} (Iteration {i})')
        fig.canvas.draw()
        if i == len(values_list) - 1:
            fig.savefig(f'Output_jpgs/{fname}_ValueIteration_Values_{i}_{AUTHOR}.jpg')
        plt.pause(interval)

    plt.ioff()
    plt.show()


def plot_policy(policy_mat: np.ndarray, fname: str, name: str = AUTHOR):
    rows, cols = policy_mat.shape
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.grid(True)
    for (y, x), symbol in np.ndenumerate(policy_mat):
        ax.text(x, y, symbol, ha='center', va='center', fontsize=12)
    plt.title(f'Policy: {name}')
    plt.savefig(f'Output_jpgs/{fname}_ValueIteration_Policy_{AUTHOR}.jpg')
    plt.show()


def plot_evaluation_sweeps(sweeps_list, fname: str, name: str = AUTHOR):
    if not sweeps_list:
        return
    iterations = np.arange(1, len(sweeps_list) + 1)
    plt.figure()
    plt.plot(iterations, sweeps_list, marker='o', linestyle='-')
    plt.xlabel('Policy-iteration #')
    plt.ylabel('Simplified value-iteration sweeps')
    plt.title(f'Policy-evaluation convergence per iteration: {AUTHOR}')
    plt.grid(True)
    out_path = f'Output_jpgs/{fname}_PolicyIteration_EvalSweeps_{AUTHOR}.jpg'
    plt.savefig(out_path, bbox_inches='tight')
    plt.show()
    plt.close()