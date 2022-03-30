import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

import myenv


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='result/optimization_results.pickle', help='Evaluation results')
    args = parser.parse_args()

    # Setup
    generations = [0, 10, 20, 30, 40, 50, 100, 150, 200, 300, 400, "final"]
    generation_max = 500

    # Create a figure
    fig = plt.figure(figsize=(6, 5))
    ax  = plt.axes([0.15, 0.15, 0.6,  0.6*6/5])
    ax2 = plt.axes([0.82, 0.15, 0.05, 0.6*6/5])

    # Load a data
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)

    # Plot the data
    for i in generations:
        results = evaluate(data, i)
        if i == "final":
            i = 500
        color = cm.get_cmap("rainbow_r")(i/generation_max)
        fig = plot_traj(results, data["task"], fig, ax, color)

    # Make a color bar
    gradient = np.linspace(0, 1, 100)
    gradient = np.vstack((gradient, gradient)).T
    ax2.pcolor(gradient, cmap="rainbow_r", vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks(np.linspace(0, 100, 6))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yticklabels([str(int(x)) for x in np.linspace(0, generation_max, 6)])
    ax2.set_ylabel("Number of generations")

    # Setup figure layout
    if data["task"] == "horizontal":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim((-0.20, 0.20))
        ax.set_yticks(np.linspace(-0.20, 0.20, 5))
    elif data["task"] == "vertical":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim(( 0.05, 0.45))
        ax.set_yticks(np.linspace(0.05, 0.45, 5))
    elif data["task"] == "facing":
        ax.set_xlabel("$y$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_xlim((-0.20, 0.20))
        ax.set_ylim(( 0.05, 0.45))
        ax.set_yticks(np.linspace(0.05, 0.45, 5))
    elif data["task"] == "upper":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim((-0.20, 0.20))
        ax.set_yticks(np.linspace(-0.20, 0.20, 5))
    fig.tight_layout()

    # Save the figure
    out_dir = "result"
    figname = "trajectory_progress.pdf"
    fig.savefig(os.path.join(out_dir, figname))
    plt.show()


def plot_traj(data, task, fig, ax, color="C0"):
    # Extract variables
    t           = data["t"]
    finger_pos  = data["finger_pos"]
    target_pos  = data["observation"][:, 0, 14:17]
    center_pos  = np.array([0.21, 0.0, 0.24])
    n_rollouts = t.shape[0]
    t_target = 49
    t_finish = 99

    # Setup axes
    if task == "horizontal" or task == "upper":
        x = 0
        y = 1
        z = 2
    elif task == "vertical":
        x = 0
        y = 2
        z = 1
    elif task == "facing":
        x = 1
        y = 2
        z = 0

    for n in range(n_rollouts):
        # Plot
        ax.scatter(target_pos[n, x], target_pos[n, y], marker='x', color='black')  # target position
        ax.scatter(center_pos[x], center_pos[y], marker='o', color='black')  # finish position
        ax.scatter(finger_pos[n, t_target, x], finger_pos[n, t_target, y], marker='x', color=color)
        ax.scatter(finger_pos[n, t_finish, x], finger_pos[n, t_finish, y], marker='o', color=color)
        ax.plot(finger_pos[n, :, x], finger_pos[n, :, y], lw=1, color=color)

    return fig


def evaluate(data, generation):
    results = {
        "t": [],
        "observation": [],
        "finger_pos": [],
    }

    for i in range(len(data["envs"])):
        # Add new sub-lists to the record data
        for k in results.keys():
            results[k].append([])

        # Load a deterministic environment
        env = data["envs"][i]

        # Get an action sequence
        activities = data["activities"][i][generation]
        X = data["model"].decode(activities)  # shape: (1, T, n_act)

        # Execute a trial
        env.alpha1 = 0.002
        env.alpha2 = 0.2
        obs = env.reset()
        r_total = 0.0
        done = False
        cnt = 0
        while not done:
            action = X[0, cnt, :]

            # Record the current states
            results["t"][-1].append(env.t)
            results["observation"][-1].append(obs.tolist())
            results["finger_pos"][-1].append(env.fingertip_position.tolist())

            obs, r, done, _ = env.step(action)
            r_total += float(r)
            cnt += 1

    # Convert the record to ndarray
    for k, v in results.items():
        results[k] = np.array(v)

    return results


if __name__ == '__main__':
    main()
