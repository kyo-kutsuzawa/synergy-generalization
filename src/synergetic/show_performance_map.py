import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a performance map of optimization results.")
    parser.add_argument("--targets", choices=["horizontal", "sagittal", "frontal", "upper"], default="horizontal")
    args = parser.parse_args()

    if args.targets == "horizontal":
        target_char = "h"
    elif args.targets == "sagittal":
        target_char = "s"
    elif args.targets == "frontal":
        target_char = "f"
    elif args.targets == "upper":
        target_char = "u"

    n_trials = 3  # Number of trials
    n_min = 0  # Min number of motor synergies
    n_max = 9  # Max number of motor synergies
    filedir = "result/optim_{}{}/result_{}_{}/optimization_results.pickle"

    # Setup file information (not every file should exist)
    fileinfo = []
    for n in range(1, n_trials + 1):
        for i in range(n_max):
            for j in range(n_max):
                fileinfo.append([filedir.format(target_char, n, i, j), i, j])

    # Setup score variables
    score = np.full((n_max, n_max), 100.0)
    score_std = np.full((n_max, n_max), 100.0)

    # Load result files
    results = load_results(fileinfo)

    # Compute the reaching accuracy
    for data, i, j in results:
        score[i, j], score_std[i, j] = compute_accuracy(data)

    print(score)
    score_max = max([score[i, j] for _, i, j in results])
    score_min = min([score[i, j] for _, i, j in results])
    score_max = 16.0
    score_min =  0.0

    # Plot accuracy
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)

    ax = plt.axes([0.1, 0.1, 0.92, 0.82])
    im = ax.pcolor(score, cmap="viridis_r", vmin=score_min, vmax=score_max)
    ax.set_xticks(np.arange(n_max) + 0.5, minor=False)
    ax.set_yticks(np.arange(n_max) + 0.5, minor=False)
    ax.set_xticklabels([str(i) for i in range(n_max)], minor=False)
    ax.set_yticklabels([str(i) for i in range(n_max)], minor=False)
    ax.set_xlim(xmin=n_min)
    ax.set_ylim(ymin=n_min)
    fig.colorbar(im, ax=ax)

    for data, i, j in results:
        if score[i, j] < 10:
            ax.text(j+0.5, i+0.5, "${:.3f}$\n$\pm{:.2f}$".format(score[i, j], score_std[i, j]), fontsize=11, horizontalalignment="center", verticalalignment="center")
        else:
            ax.text(j+0.5, i+0.5, "${:.2f}$\n$\pm{:.2f}$".format(score[i, j], score_std[i, j]), fontsize=11, horizontalalignment="center", verticalalignment="center")

    # Text for 0 synergies
    ax.text(0.5, 0.5, "N/A", fontsize=11, horizontalalignment="center", verticalalignment="center")

    # Setup figure layout
    ax.set_xlabel("$L_\\mathrm{s}$", fontsize=14)
    ax.set_ylabel("$L_\\mathrm{h}$", fontsize=14)

    # Save the figure
    figname = "result/performance_map.pdf"
    fig.savefig(figname)

    # Show the figure
    plt.show()


def load_results(fileinfo):
    results = {}
    keys = ["finger_pos", "observation"]

    for filedir, i, j in fileinfo:
        # Skip if the file does not exist
        if not os.path.exists(filedir):
            continue

        # Open the file
        with open(filedir, 'rb') as f:
            data = pickle.load(f)

        if (i, j) in results:
            data_old = results[i, j]
            for key in keys:
                data[key] = np.concatenate((data_old[key], data[key]), axis=0)

        results[i, j] = data

    ret = [(data, i, j) for (i, j), data in results.items()]

    return ret


def compute_accuracy(data):
    # Extract variables
    t_target = 50
    finger_pos = data["finger_pos"][:, t_target, :]
    target_pos = data["observation"][:, 0, 14:17]

    dev = target_pos - finger_pos
    errors = np.sqrt(np.sum(np.square(dev), axis=1))

    return np.mean(errors) * 100, np.std(errors) * 100


if __name__ == "__main__":
    main()
