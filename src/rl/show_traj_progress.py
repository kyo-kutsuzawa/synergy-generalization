import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12


def generate(filename, figname):
    # List of file names
    filelist = [(i, filename.format(i)) for i in (0, 10, 20, 30, 50, 100, 150, 200, 250, 300)]
    iteration_max = 300000

    # Create a figure
    fig = plt.figure(figsize=(6, 5))
    ax  = plt.axes([0.15, 0.15, 0.6,  0.6*6/5])
    ax2 = plt.axes([0.82, 0.15, 0.05, 0.6*6/5])

    for i, filename in filelist:
        # Load a data
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Plot the data
        color = cm.get_cmap("rainbow_r")(i/(iteration_max/1e3))
        fig = plot_traj(data, fig, ax, color)

    # Make a color bar
    gradient = np.linspace(0, 1, 100)
    gradient = np.vstack((gradient, gradient)).T
    ax2.pcolor(gradient, cmap="rainbow_r", vmin=0, vmax=1)
    ax2.set_xticks([])
    ax2.set_yticks(np.linspace(0, 100, 6))
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_yticklabels([str(int(x)) for x in np.linspace(0, iteration_max/1e4, 6)])
    ax2.set_ylabel("Number of trials ($\\times 10^{4}$)")

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
    fig.tight_layout()

    # Save the figure
    out_dir = "result"
    fig.savefig(os.path.join(out_dir, figname))
    plt.show()


def plot_traj(data, fig, ax, color="C0"):
    # Extract variables
    t           = data["t"]
    finger_pos  = data["finger_pos"]
    target_pos  = data["observation"][:, 0, 14:17]
    center_pos  = np.array([0.21, 0.0, 0.24])
    n_rollouts = t.shape[0]
    t_target = 50
    t_finish = 100

    # Setup axes
    if data["task"] == "horizontal":
        x = 0
        y = 1
        z = 2
    elif data["task"] == "vertical":
        x = 0
        y = 2
        z = 1
    elif data["task"] == "facing":
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


if __name__ == '__main__':
    generate("result/result_h_save/result{}.pickle", "trajectory_progress-horizontal.pdf")
    generate("result/result_v_save/result{}.pickle", "trajectory_progress-sagittal.pdf")
