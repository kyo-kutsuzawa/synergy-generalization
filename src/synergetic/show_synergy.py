import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import synergy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.left'] = False
plt.rcParams['axes.titlesize'] = 'medium'


def main1():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='result/resultA1.pickle', help='Movement dataset')
    parser.add_argument('--n-synergies', type=int, default=3, help='Number of synergies')
    parser.add_argument('--out', type=str, default='result', help='Output directory')
    args = parser.parse_args()

    # Load recorded actions
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    t = data['t']  # shape: (n_trials, length, n_dof)
    actions = data['action']  # shape: (n_trials, length, n_dof)

    # Extract motor synergies
    synergy_model = synergy.SpatioTemporalSynergy(n_synergies=args.n_synergies, method="negative-nmf")
    synergy_model.extract(actions)
    n_dof = synergy_model.dof

    # Create a figure and axes
    fig = plt.figure(figsize=(6, 12))
    axes = []
    for k in range(args.n_synergies):
        axes.append([])
        for m in range(n_dof):
            ax = fig.add_subplot(n_dof, args.n_synergies, m*args.n_synergies+k+1)
            ax.tick_params(labelleft=False, labelbottom=False)
            axes[k].append(ax)

    # Plot data
    for m in range(n_dof):
        for k in range(args.n_synergies):
            axes[k][m].fill_between(t[0], np.zeros_like(t[0]), synergy_model.synergies[k, :, m])

    # Setup plot range
    for k in range(args.n_synergies):
        val_max = np.max(synergy_model.synergies[k, :, :]) * 1.2
        for m in range(n_dof):
            axes[k][m].set_xlim(0, 2)
            axes[k][m].set_ylim((0, val_max))

    # Setup label and ticks
    for k in range(args.n_synergies):
        axes[k][0].tick_params(labelright=True)
        axes[k][0].yaxis.set_label_position("left")
        axes[k][0].yaxis.tick_right()
        axes[k][-1].tick_params(labelbottom=True)
        axes[k][-1].set_xlabel("Time [s]")
        title = fig.add_subplot(1, 3, k+1)
        title.set_title('synergy #{}'.format(k))
        title.set_axis_off()
    for m in range(n_dof):
        if m < n_dof / 2:
            axes[0][m].set_ylabel("$a_{{+}}^{}$".format(m+1), rotation='horizontal', ha='right', va='center')
        else:
            axes[0][m].set_ylabel("$a_{{-}}^{}$".format(m+1 - int(n_dof / 2)), rotation='horizontal', ha='right', va='center')

    # Setup layout
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(os.path.join(args.out, "motor_synergies.pdf"))

    plt.show()


def main2():
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='result/resultA1.pickle', help='Movement dataset')
    parser.add_argument('--n-synergies', type=int, default=3, help='Number of synergies')
    parser.add_argument('--out', type=str, default='result', help='Output directory')
    args = parser.parse_args()

    # Load recorded actions
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    t = data['t']  # shape: (n_trials, length, n_dof)
    actions = data['action']  # shape: (n_trials, length, n_dof)

    # Extract motor synergies
    synergy_model = synergy.SpatioTemporalSynergy(n_synergies=args.n_synergies, method="negative-nmf")
    synergy_model.extract(actions)
    n_dof = synergy_model.dof

    # Create a figure and axes
    fig = plt.figure(figsize=(6, 10))
    gs_positive = GridSpec(nrows=int(n_dof/2), ncols=args.n_synergies)
    gs_negative = GridSpec(nrows=int(n_dof/2), ncols=args.n_synergies)

    axes = []
    for k in range(args.n_synergies):
        axes.append([])
        for m in range(n_dof):
            if m < n_dof / 2:
                ax = fig.add_subplot(gs_positive[m, k])
            else:
                ax = fig.add_subplot(gs_negative[m-int(n_dof/2), k])
            ax.tick_params(labelleft=False, labelbottom=False)
            axes[k].append(ax)

    # Plot data
    for m in range(n_dof):
        for k in range(args.n_synergies):
            axes[k][m].fill_between(t[0], np.zeros_like(t[0]), synergy_model.synergies[k, :, m])

    # Setup plot range
    for k in range(args.n_synergies):
        val_max = np.max(synergy_model.synergies[k, :, :]) * 1.2
        for m in range(n_dof):
            axes[k][m].set_xlim(0, 2)
            axes[k][m].set_ylim((0, val_max))

    # Setup label and ticks
    for k in range(args.n_synergies):
        axes[k][0].tick_params(labelright=True)
        axes[k][0].yaxis.set_label_position("left")
        axes[k][0].yaxis.tick_right()
        axes[k][int(n_dof/2)].tick_params(labelright=True)
        axes[k][int(n_dof/2)].yaxis.set_label_position("left")
        axes[k][int(n_dof/2)].yaxis.tick_right()
        axes[k][-1].tick_params(labelbottom=True)
        axes[k][-1].set_xlabel("Time [s]")

        title = fig.add_subplot(1, 3, k+1)
        title.set_title('synergy #{}'.format(k))
        title.set_axis_off()

    for m in range(n_dof):
        if m < n_dof / 2:
            axes[0][m].set_ylabel("#{}".format(m+1))
        else:
            axes[0][m].set_ylabel("#{}".format(m+1 - int(n_dof / 2)))

    axes[0][int(n_dof*1/4)].text(-0.8, 0.0, "Positive", rotation=90)
    axes[0][int(n_dof*3/4)].text(-0.8, 0.0, "Negative", rotation=90)

    # Setup layout
    gs_positive.tight_layout(fig)
    gs_negative.tight_layout(fig)
    gs_positive.update(bottom=0.51, top=0.9, hspace=0.05)
    gs_negative.update(top=0.49, hspace=0.05)

    # Save the figure
    plt.savefig(os.path.join(args.out, "motor_synergies.pdf"))

    plt.show()


def main3():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='result/resultA1.pickle', help='Movement dataset')
    parser.add_argument('--n-synergies', type=int, default=3, help='Number of synergies')
    parser.add_argument('--out', type=str, default='result', help='Output directory')
    args = parser.parse_args()

    # Load recorded actions
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    t = data['t'][:, :-1]  # shape: (n_trials, length)
    actions = data['action']  # shape: (n_trials, length, n_dof)

    # Extract motor synergies
    synergy_model = synergy.SpatioTemporalSynergy(n_synergies=args.n_synergies, method="negative-nmf")
    synergy_model.extract(actions)
    n_dof = int(synergy_model.dof/2)

    # Create a figure and axes
    fig = plt.figure(figsize=(6, 6))
    axes = []
    for k in range(args.n_synergies):
        axes.append([])
        for m in range(n_dof):
            ax = fig.add_subplot(n_dof, args.n_synergies, m*args.n_synergies+k+1)
            ax.tick_params(labelleft=False, labelbottom=False)
            axes[k].append(ax)

    # Plot data
    for m in range(n_dof):
        for k in range(args.n_synergies):
            axes[k][m].fill_between(t[0], np.zeros_like(t[0]), +synergy_model.synergies[k, :-1, m], color='C3')
            axes[k][m].fill_between(t[0], np.zeros_like(t[0]), -synergy_model.synergies[k, :-1, m+n_dof], color='C4')
            axes[k][m].plot([0, 1], [0, 0], lw=1, color='black')

    # Setup plot range
    val_max = np.max(synergy_model.synergies) * 1.2
    for k in range(args.n_synergies):
        #val_max = np.max(synergy_model.synergies[k, :, :]) * 1.2
        for m in range(n_dof):
            axes[k][m].set_xlim(0, 1)
            axes[k][m].set_ylim((-val_max, val_max))

    # Setup label and ticks
    for k in range(args.n_synergies):
        for m in range(n_dof):
            axes[k][m].set_xticks((0, 0.5, 1))
    for k in range(args.n_synergies):
        #axes[k][0].tick_params(labelright=True)
        #axes[k][0].yaxis.set_label_position("left")
        #axes[k][0].yaxis.tick_right()
        axes[k][-1].tick_params(labelbottom=True)
        axes[k][-1].set_xticklabels([0, "", 1])
        axes[k][-1].set_xlabel("Time [s]")
        title = fig.add_subplot(1, args.n_synergies, k+1)
        title.set_title('Synergy {}'.format(k+1))
        title.set_axis_off()
    axes[-1][0].tick_params(labelright=True)
    axes[-1][0].yaxis.set_label_position("left")
    axes[-1][0].yaxis.tick_right()
    for m in range(n_dof):
        axes[0][m].set_ylabel("Joint {}".format(m+1), rotation='horizontal', ha='right', va='center')

    # Setup layout
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)

    # Save the figure
    plt.savefig(os.path.join(args.out, "motor_synergies.pdf"))

    plt.show()


if __name__ == '__main__':
    main3()
