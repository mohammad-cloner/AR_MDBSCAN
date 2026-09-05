# AR-MDBSCAN — Adaptive Reachability-density Multi-Density DBSCAN

A multi-density, density-based clustering framework with a **single user-tunable
parameter** (the density threshold `t`). This repository contains the clean,
standalone reference implementation of the clustering method proposed in the
accompanying doctoral thesis.

The framework extends the ideas of DBSCAN and MDBSCAN with three mechanisms:

1. **Adaptive DBSCAN per region** — for every subset, `eps` is taken from the
   knee point of the sorted k-distance curve and `MinPts` is derived from the
   subset size and dimensionality (fully data-driven).
2. **Geometric attachment test** — instead of size-based criteria, a component
   of the low-density region is judged by its *attachment fraction*: the share
   of its points that lie within `r` × local-scale of the nearest high-density
   anchors. Attached components are fringe/background; detached components
   become independent clusters.
3. **Rescue pass with scale validation** — fringe points and leftover noise are
   re-clustered; a component is accepted as a genuine cluster only when its
   internal scale is small relative to its distance to the existing clusters.
   The result is a **complete partition** (no unlabelled points).

No metaheuristic (GA/PSO/RL) is used anywhere in the code: the threshold `t` is
supplied by the caller, and all remaining parameters are data-driven or fixed
design constants.

## How it works

```
X ──► distance matrix + k = ⌊log₂ n⌋ neighbours
  ──► reachability-based density metric ρ
  ──► split by threshold t  →  L (low) / H (high)
  ──► adaptive DBSCAN on L  →  components + noise
        ├── attachment test (r, τ):  attached → fringe,  detached → cluster
        └── point-wise fringe/insignificant split
  ──► adaptive DBSCAN on H ∪ insignificant
  ──► rescue pass (β) over fringe + HD-noise  →  genuine small clusters
  ──► assign leftovers to nearest labelled point  →  complete partition
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.8 with `numpy`, `scipy` and `scikit-learn`
(`matplotlib` only for the optional `--plot` flag).

## Quick start

```bash
# compound benchmark, threshold t = 4.86
python examples/run_example.py --data datasets/compound.mat --t 4.86

# Iris (real data), threshold t = 14.94, save a plot
python examples/run_example.py --data datasets/iris.mat --t 14.94 --plot iris.png
```

### Python API

```python
from ar_mdbscan import ar_mdbscan, load_dataset

X, y = load_dataset("datasets/compound.mat")   # y may be None
labels = ar_mdbscan(X, t=4.86)                 # complete partition, no noise
```

### Reproducing the thesis results

`examples/reproduce_thesis.py` runs the method on all eleven benchmark
datasets with the thresholds reported in the thesis and prints the resulting
NMI / ARI / cluster counts, which match the thesis tables:

```bash
python examples/reproduce_thesis.py
```

## Parameters

| Parameter | Role | Default | Nature |
|-----------|------|---------|--------|
| `t`       | density threshold splitting low/high regions | — (required) | **user-tunable** |
| `r`       | attachment scale factor in the attachment test | `2.0` | fixed design constant |
| `tau`     | attachment-fraction accept/reject threshold | `0.5` | fixed design constant |
| `beta`    | internal/external scale factor in the rescue pass | `0.35` | fixed design constant |
| `k`       | neighbour count for the density metric | `⌊log₂ n⌋` | data-driven |
| `MinPts`, `eps` | per-subset DBSCAN parameters | `max(3, ⌊log₂ nₛ⌋, d+1)`, knee of k-distance curve | data-driven |

## Datasets

The `datasets/` folder contains the eleven benchmark datasets used in the
thesis experiments (8 synthetic 2-D benchmarks + 3 real UCI datasets), with
the threshold `t` reported in the thesis for each:

| Dataset | Samples | Classes | `t` (thesis) | Source |
|---------|---------|---------|--------------|--------|
| compound | 399 | 6 | 4.86 | standard 2-D clustering benchmark |
| aggregation | 788 | 7 | 5.67 | Karypis et al. benchmark |
| crossline | 1000 | 2 | 320 | synthetic benchmark |
| cure-t2-4k | 4200 | 7 | 535 | CURE-based benchmark |
| D31 | 3100 | 31 | 20.1 | Veenman et al. benchmark |
| filled_circle | 500 | 4 | 19.5 | nested-rings synthetic dataset |
| flame | 240 | 2 | 5.76 | standard 2-D clustering benchmark |
| jain | 373 | 2 | 4.66 | Jain & Law (2005) |
| iris | 150 | 3 | 14.94 | UCI Machine Learning Repository |
| seeds | 210 | 3 | 8.80 | UCI Machine Learning Repository |
| thyroid | 215 | 3 | 0.41 | UCI Machine Learning Repository |

## Notes on complexity

The current implementation materialises the full `n × n` Euclidean distance
matrix, matching the complexity class of classic DBSCAN with pre-computed
distances: **O(n²)** time and memory. It is therefore intended for datasets up
to a few hundred thousand points; approximate nearest-neighbour structures are
a natural extension for larger scales.

## Project structure

```
ar-mdbscan/
├── ar_mdbscan/
│   ├── algorithm.py      # core AR-MDBSCAN pipeline
│   └── data_loader.py    # .mat / text dataset loading
├── datasets/             # the 11 benchmark datasets used in the thesis
├── examples/
│   └── run_example.py    # command-line demo
├── requirements.txt
└── README.md
```

## Citation

If you use this implementation, please cite the accompanying doctoral thesis
(on the AR-MDBSCAN multi-density clustering framework).

## License

Released under the MIT License — see [LICENSE](LICENSE).
