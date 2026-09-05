# -*- coding: utf-8 -*-
"""Reproduce the thesis chapter-4 results with the clean AR-MDBSCAN package.

Runs the method on all eleven benchmark datasets using the density threshold
``t`` reported in the thesis for each dataset, and prints NMI / ARI / cluster
count / noise side by side with the reference values from the thesis tables.

Usage
-----
python examples/reproduce_thesis.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ar_mdbscan import ar_mdbscan, load_dataset
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

# dataset file -> (thesis threshold t, thesis NMI, thesis ARI, thesis #clusters)
THESIS_RESULTS = {
    "compound.mat":            (4.86, 1.0000, 1.0000, 6),
    "aggregation.mat":         (5.67, 0.9915, 0.9949, 7),
    "crossline.mat":           (320.0, 0.3997, 0.2153, 12),
    "cure-t2-4k.mat":          (535.0, 0.9818, 0.9870, 7),
    "D31.mat":                 (20.1, 0.9661, 0.9510, 31),
    "filled_circle_2_500.csv": (19.5, 0.8514, 0.8000, 8),
    "flame.mat":               (5.76, 0.9269, 0.9666, 2),
    "jain.mat":                (4.66, 1.0000, 1.0000, 2),
    "iris.mat":                (14.94, 0.7980, 0.7455, 3),
    "seeds.mat":               (8.80, 0.6552, 0.6496, 4),
    "thyroid.mat":             (0.41, 0.5951, 0.6957, 3),
}


def main() -> None:
    header = "%-24s %8s %8s %8s %8s %4s %6s" % (
        "dataset", "NMI", "(thesis)", "ARI", "(thesis)", "K", "noise%")
    print(header)
    print("-" * len(header))

    for filename, (t, ref_nmi, ref_ari, ref_k) in THESIS_RESULTS.items():
        X, y = load_dataset(os.path.join(DATASETS_DIR, filename))
        labels = ar_mdbscan(X, t=t)

        nmi = normalized_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        k = len(set(labels))
        noise_pct = 100.0 * sum(labels < 0) / len(labels)
        print("%-24s %8.4f %8.4f %8.4f %8.4f %4d %6.1f"
              % (filename, nmi, ref_nmi, ari, ref_ari, k, noise_pct))


if __name__ == "__main__":
    main()
