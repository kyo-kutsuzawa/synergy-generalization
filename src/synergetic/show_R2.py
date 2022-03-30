import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import synergy

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12


def main():
    """Evaluate reconstraction performance, changing the number of motor synergies.
    """
    import argparse

    # Process commandline arguments
    parser = argparse.ArgumentParser(description="Compute R2 values, changing the number of motor synergies.")
    parser.add_argument('--filename', type=str, default='result/resultA1.pickle', help='')
    parser.add_argument('--max', type=int, default=8, help='Maximum number of motor synergies')
    parser.add_argument('--out', type=str, default='result', help='Directory name where results to be saved.')
    args = parser.parse_args()

    # Load recorded actions
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    actions = data['action']  # shape: (n_trials, length, n_dof)

    # Extract synergies while changing the number of motor synergies
    errors = []
    for n in range(args.max):
        # Create a synergy model
        model = synergy.SpatioTemporalSynergy(n_synergies=n+1, method="negative-nmf")
        model.extract(actions, max_iter=10000)

        # Compute synergy activities and reconstruction actions
        activities = model.encode(actions)
        actions_reconstruct = model.decode(activities)
        R2_value = synergy.R2(actions, actions_reconstruct)
        errors.append(R2_value)
        print("{} motor synergies: R2={}".format(n+1, R2_value))

    # Plot the results
    plt.figure(figsize=(6, 4))
    plt.plot([1, args.max], [0.95, 0.95], ls=":", color="black")
    plt.plot(np.arange(args.max)+1, errors, lw=2, marker="o", color="C0")
    plt.xlim(xmin=1, xmax=args.max)
    plt.ylim(ymin=0.5, ymax=1.0)
    plt.xlabel("Number of motor synergies")
    plt.ylabel("$R^2$")
    plt.tight_layout()

    plt.savefig(os.path.join(args.out, "R2.pdf"))
    plt.show()


if __name__ == '__main__':
    main()
