"""This script evaluates how synergy activities vary with the direction of reaching movements.
"""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import synergy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='result/resultA1.pickle', help='Movement dataset')
    parser.add_argument('--filename2', type=str, default=None, help='Movement dataset (not necessary)')
    parser.add_argument('--n-synergies', type=int, default=3, help='Number of synergies')
    parser.add_argument('--out', type=str, default='result', help='Output directory')
    args = parser.parse_args()

    # Load recorded actions
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    actions = data['action']  # shape: (n_trials, length, n_dof)

    # Extract motor synergies
    synergy_model = synergy.SpatioTemporalSynergy(n_synergies=args.n_synergies, method="negative-nmf")
    synergy_model.extract(actions)
    activities = synergy_model.encode(actions)

    # To close the radar chart plots, append the first sample to activities again
    activities = np.concatenate((activities, activities[[0], :]), axis=0)

    # Load the task name
    task = data['task']

    # Plot synergy activities on a radar chart
    plt.figure(figsize=(6, 4)).add_subplot(111, polar=True)
    target_params = np.linspace(-0.5*np.pi, 1.5*np.pi, num=activities.shape[0], endpoint=True)[::-1]
    for i in range(args.n_synergies):
        #plt.plot(target_params, activities[:, i], label="synergy #{}".format(i))
        plt.plot(target_params, activities[:, i], label="Synergy {}".format(i+1), ls="", marker="o")

    if args.filename2 is not None:
        with open(args.filename2, 'rb') as f:
            data2 = pickle.load(f)
        activities = synergy_model.encode(data2['action'])
        activities = np.concatenate((activities, activities[[0], :]), axis=0)

        target_params = np.linspace(-0.5*np.pi, 1.5*np.pi, num=activities.shape[0], endpoint=True)[::-1]
        for i in range(args.n_synergies):
            plt.plot(target_params, activities[:, i], label="_nolegend_", ls="-", color="C{}".format(i))

    plt.yticks(np.linspace(0, 0.8, 5))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(args.out, "synergy_direction.pdf"))

    plt.show()


if __name__ == "__main__":
    main()
