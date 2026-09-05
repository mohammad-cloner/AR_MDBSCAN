# -*- coding: utf-8 -*-
"""Command-line example: run AR-MDBSCAN on a dataset.

Usage
-----
python examples/run_example.py --data datasets/compound.mat --t 4.8636
python examples/run_example.py --data datasets/iris.mat --t 14.9415 --plot out.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ar_mdbscan import ar_mdbscan, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AR-MDBSCAN (multi-density density-based clustering).")
    parser.add_argument("--data", required=True,
                        help="path to the dataset (.mat/.txt/.csv/.arff)")
    parser.add_argument("--t", type=float, required=True,
                        help="density threshold (the single user-tunable parameter)")
    parser.add_argument("--r", type=float, default=2.0,
                        help="attachment scale factor (default: 2.0)")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="attachment-fraction threshold (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.35,
                        help="rescue validation factor (default: 0.35)")
    parser.add_argument("--plot", default=None,
                        help="optional output path for a scatter plot of the result")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"dataset : {args.data}")
    print(f"samples : {X.shape[0]}   features: {X.shape[1]}")
    if y is not None:
        print(f"classes : {len(np.unique(y))}")

    labels = ar_mdbscan(X, t=args.t, r=args.r, tau=args.tau, beta=args.beta)

    n_clusters = len(np.unique(labels))
    n_noise = int(np.sum(labels < 0))
    print(f"clusters: {n_clusters}")
    print(f"noise   : {n_noise} ({100.0 * n_noise / len(labels):.1f}%)")

    if y is not None:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        print(f"NMI     : {normalized_mutual_info_score(y, labels):.4f}")
        print(f"ARI     : {adjusted_rand_score(y, labels):.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        if X.shape[1] >= 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=18, cmap="tab20")
        else:
            plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=labels, s=18, cmap="tab20")
        plt.title(f"AR-MDBSCAN (t={args.t}) - {n_clusters} clusters")
        plt.savefig(args.plot, dpi=200, bbox_inches="tight")
        print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
