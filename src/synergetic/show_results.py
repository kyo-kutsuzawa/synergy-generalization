import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12


def main():
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description='Show evaluation results')
    parser.add_argument('--filename', type=str, default='result/training_results.pickle', help='Evaluation results')
    parser.add_argument('--visualize', '-v', nargs='+', choices=['all', 'pos', 'pos2d', 'pos3d', 'activity', 'activity-polar', 'err', 'video'], default='all')
    parser.add_argument('--out', type=str, default='result', help='Directory name where results to be saved.')
    args = parser.parse_args()

    # Load evaluation results
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)

    if 'pos' in args.visualize or 'all' in args.visualize:
        plot_hand_positions(data)

    if 'pos2d' in args.visualize or 'all' in args.visualize:
        plot_hand_positions2d(data, args.out)

    if 'pos3d' in args.visualize or 'all' in args.visualize:
        plot_hand_positions3d(data, args.out)

    if 'activity' in args.visualize or 'all' in args.visualize:
        plot_activity(data, args.out)

    if 'activity-polar' in args.visualize or 'all' in args.visualize:
        plot_activity_polar(data, args.out)

    if 'err' in args.visualize or 'all' in args.visualize:
        compute_errors(data, args.out)

    if 'video' in args.visualize or 'all' in args.visualize:
        record_video(data, args.out)

    plt.show()


def plot_hand_positions(data):
    # Create a figure
    fig = plt.figure()

    # Extract variables
    t           = data["t"]
    finger_pos  = data["finger_pos"]
    target_pos  = data["observation"][:, :, 14:17]
    n_rollouts = t.shape[0]

    # Plot target positions
    for n in range(n_rollouts):
        plt.plot(t[n], target_pos[n, :, 0], ls='--', color='C0')  # Target x
        plt.plot(t[n], target_pos[n, :, 1], ls='--', color='C1')  # Target y
        plt.plot(t[n], target_pos[n, :, 2], ls='--', color='C2')  # Target z

    # Plot finger positions
    for n in range(n_rollouts):
        plt.plot(t[n], finger_pos[n, :, 0], color='C0')  # Finger x
        plt.plot(t[n], finger_pos[n, :, 1], color='C1')  # Finger y
        plt.plot(t[n], finger_pos[n, :, 2], color='C2')  # Finger z

    return fig


def plot_hand_positions2d(data, out_dir):
    # Create a figure
    fig = plt.figure(figsize=(6, 5))
    ax  = plt.axes([0.13, 0.15, 0.6,  0.6*6/5])
    ax2 = plt.axes([0.81, 0.15, 0.15, 0.6*6/5])

    # Extract variables
    t           = data["t"]
    finger_pos  = data["finger_pos"]
    target_pos  = data["observation"][:, 0, 14:17]
    center_pos  = np.array([0.21, 0.0, 0.24])
    n_rollouts = t.shape[0]
    t_target =  50
    t_finish = 100

    # Setup axes
    if data["task"] == "horizontal" or data["task"] == "upper":
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
        l1  = ax.scatter(target_pos[n, x], target_pos[n, y], marker='x', color='black')  # target position
        l2  = ax.scatter(center_pos[x], center_pos[y], marker='o', color='black')  # finish position
        l3  = ax.scatter(finger_pos[n, t_target, x], finger_pos[n, t_target, y], marker='x', color='C0')
        l4  = ax.scatter(finger_pos[n, t_finish, x], finger_pos[n, t_finish, y], marker='o', color='C0')
        l5, = ax.plot(finger_pos[n, :, x], finger_pos[n, :, y], color='C0')

        ax2.scatter(target_pos[n, z], target_pos[n, y], marker='x', color='black')  # target position
        ax2.scatter(center_pos[z], center_pos[y], marker='o', color='black')  # finish position
        ax2.scatter(finger_pos[n, t_target, z], finger_pos[n, t_target, y], marker='x', color='C0')
        ax2.scatter(finger_pos[n, t_finish, z], finger_pos[n, t_finish, y], marker='o', color='C0')
        ax2.plot(finger_pos[n, :, z], finger_pos[n, :, y], color='C0')

        # Register labels
        if n == 0:
            handles = [l5, l1, l3, l2, l4]
            labels  = ["trajectory", "$p_\\mathrm{target}$", "$p(t_\\mathrm{target})$", "$p_\\mathrm{finish}$", "$p(t_\\mathrm{finish})$"]

    # Setup figure layout
    if data["task"] == "horizontal":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim((-0.20, 0.20))
        ax.set_yticks(np.linspace(-0.20, 0.20, 5))
        ax2.set_xlabel("$z$ [m]")
        ax2.set_xlim(( 0.19, 0.29))
        ax2.set_ylim((-0.20, 0.20))
        ax2.set_xticks((0.19, 0.29))
    elif data["task"] == "vertical":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim(( 0.05, 0.45))
        ax.set_yticks(np.linspace(0.05, 0.45, 5))
        ax2.set_xlabel("$y$ [m]")
        ax2.set_xlim((-0.05, 0.05))
        ax2.set_ylim(( 0.05, 0.45))
        ax2.set_xticks((-0.05, 0.05))
    elif data["task"] == "facing":
        ax.set_xlabel("$y$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_xlim((-0.20, 0.20))
        ax.set_ylim(( 0.05, 0.45))
        ax.set_yticks(np.linspace(0.05, 0.45, 5))
        ax2.set_xlabel("$x$ [m]")
        ax2.set_xlim((0.16, 0.26))
        ax2.set_ylim(( 0.05, 0.45))
        ax2.set_xticks((0.16, 0.26))
    elif data["task"] == "upper":
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_xlim((-0.00, 0.40))
        ax.set_ylim((-0.20, 0.20))
        ax.set_yticks(np.linspace(-0.20, 0.20, 5))
        ax2.set_xlabel("$z$ [m]")
        ax2.set_xlim(( 0.22, 0.32))
        ax2.set_ylim((-0.20, 0.20))
        ax2.set_xticks((0.22, 0.32))
    ax2.set_yticks([])
    fig.tight_layout()

    # Save the figure
    figname = "trajectories_synergetic.pdf"
    fig.savefig(os.path.join(out_dir, figname))

    return fig


def plot_hand_positions3d(data, out_dir):
    # Create a figure
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8], projection='3d')

    # Extract variables
    t           = data["t"]
    finger_pos  = data["finger_pos"]
    target_pos  = data["observation"][:, 0, 14:17]
    center_pos  = np.array([0.3, 0.0, 0.6-0.25])
    n_rollouts = t.shape[0]
    t_target =  50
    t_finish = 100

    for n in range(n_rollouts):
        # Plot
        l1  = ax.scatter(target_pos[n, 0], target_pos[n, 1], target_pos[n, 2], marker='x', color='black')  # target position
        l2  = ax.scatter(center_pos[0], center_pos[1], center_pos[2], marker='o', color='black')  # finish position
        l3  = ax.scatter(finger_pos[n, t_target, 0], finger_pos[n, t_target, 1], finger_pos[n, t_target, 2], marker='x', color='C1')
        l4  = ax.scatter(finger_pos[n, t_finish, 0], finger_pos[n, t_finish, 1], finger_pos[n, t_finish, 2], marker='o', color='C1')
        l5, = ax.plot(finger_pos[n, :, 0], finger_pos[n, :, 1], finger_pos[n, :, 2], color='C0')

        # Register labels
        if n == 0:
            handles = [l5, l1, l3, l2, l4]
            labels  = ["trajectory", "$p_\\mathrm{target}$", "$p(t_\\mathrm{target})$", "$p_\\mathrm{finish}$", "$p(t_\\mathrm{finish})$"]

    # Setup figure layout
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_zlabel("$z$ [m]")
    ax.set_xlim(( 0.10, 0.50))
    ax.set_ylim((-0.20, 0.20))
    ax.set_zlim(( 0.15, 0.55))
    ax.legend(handles, labels)
    fig.tight_layout()

    # Save the figure
    figname = "trajectories3d_synergetic.pdf"
    fig.savefig(os.path.join(out_dir, figname))

    return fig


def plot_activity(data, out_dir):
    # Create a figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # Extract variables
    activity = data["activity"]
    n_rollouts  = activity.shape[0]
    n_synergies = activity.shape[1]

    # Plot
    for k in range(n_synergies):
        x = np.linspace(0, 2*np.pi, n_rollouts)
        ax.plot(x, activity[:, k], label=str(k))

    fig.legend()
    fig.tight_layout()

    # Save the figure
    figname = "activity_synergetic.pdf"
    fig.savefig(os.path.join(out_dir, figname))

    return fig


def plot_activity_polar(data, out_dir):
    # Create a figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Extract variables
    activity = data["activity"]
    n_rollouts  = activity.shape[0]
    n_synergies = activity.shape[1]

    # To close the radar chart plots, append the first sample to `activity` again
    activity = np.concatenate((activity, activity[[0], :]), axis=0)

    # Plot the synergy activity on a radar chart
    target_params = np.linspace(-0.5*np.pi, 1.5*np.pi, num=n_rollouts+1, endpoint=True)[::-1]
    for i in range(n_synergies):
        plt.plot(target_params, activity[:, i], label="synergy #{}".format(i))
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(out_dir, "synergy_direction.pdf"))

    return fig


def compute_errors(data, out_dir):
    # Extract variables
    finger_pos = data["finger_pos"]
    target_pos = data["observation"][:, 0, 14:17]
    center_pos  = np.array([0.21, 0.0, 0.24])
    t_target =  50
    t_finish = 100

    # Compute errors in target position
    dev = target_pos - finger_pos[:, t_target, :]
    errors_target = np.sqrt(np.sum(np.square(dev), axis=1))
    print("Target error:   mean={}, std={}".format(np.mean(errors_target), np.std(errors_target)))

    # Compute errors in terminal position
    dev = center_pos - finger_pos[:, t_finish, :]
    errors_terminal = np.sqrt(np.sum(np.square(dev), axis=1))
    print("Terminal error: mean={}, std={}".format(np.mean(errors_terminal), np.std(errors_terminal)))

    # Print a formatted text
    fmt = lambda e: e * 100
    print("${:.3f} \pm {:.3f}$ cm & ${:.3f} \pm {:.3f}$ cm".format(fmt(np.mean(errors_target)), fmt(np.std(errors_target)), fmt(np.mean(errors_terminal)), fmt(np.std(errors_terminal))))

    return


def record_video(data, out_dir):
    import myenv

    n_rollouts = data["t"].shape[0]

    for n in range(n_rollouts):
        # Create an environment
        env = myenv.ArmReachingDeterministic(target_position=np.zeros(3))
        env.record_setup(width=1920, height=1080)
        env.target_position = data["observation"][n, 0, 14:17]

        # perform a rollout
        env.reset()
        done = False
        while not done:
            action = data["action"][n, env.cnt, :]
            _, _, done, _ = env.step(action)
            env.record()
        env.save_video("result/video{}.mp4".format(n))


if __name__ == '__main__':
    main()
