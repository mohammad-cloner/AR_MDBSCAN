# -*- coding: utf-8 -*-
"""
AR-MDBSCAN: Adaptive Reachability-density Multi-Density DBSCAN.

A multi-density, density-based clustering framework with a single
user-tunable parameter (the density threshold ``t``).

Pipeline (see the accompanying thesis, Chapter 3, for full details):

1.  Build the global Euclidean distance matrix and derive the number of
    neighbours ``k = floor(log2(n))`` (clamped to ``[3, n-1]``).
2.  Compute the reachability-based local density metric ``rho`` for every
    point (a robust variant of the standard Local Reachability Density).
3.  Split the dataset into a low-density set ``L`` and a high-density set
    ``H`` using the user-supplied threshold ``t``.
4.  Run an *adaptive* DBSCAN on ``L``: for each subset the radius ``eps``
    is taken from the knee point of the sorted k-distance curve and
    ``MinPts = max(3, floor(log2(n_s)), d + 1)``.
5.  Judge every low-density component with the *attachment fraction* test:
    a component whose points are mostly close (relative to the local scale
    of the nearest high-density anchors) is fringe/background, otherwise it
    is accepted as an independent cluster.  Individual leftover points are
    split into *fringe* points (attached to a high-density anchor) and
    *insignificant* points.
6.  Run adaptive DBSCAN on ``H`` united with the insignificant points.
7.  *Rescue pass*: fringe points and high-density noise are re-clustered;
    a component is accepted as a genuine cluster only when its internal
    scale is small relative to its distance to the existing clusters.
8.  Any remaining unlabelled point is assigned to its nearest labelled
    neighbour, producing a complete partition (no unlabelled points).

The design constants ``r`` (attachment scale factor), ``tau`` (attachment
fraction threshold) and ``beta`` (rescue validation factor) are fixed
hyper-parameters of the method (see thesis Table 3-3); they are *not*
tuned per dataset.

This module intentionally contains **no metaheuristic component**: the
threshold ``t`` is supplied by the caller.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

__all__ = ["ar_mdbscan", "reachability_density", "find_knee_point", "adaptive_dbscan"]

# Default design constants (fixed across all experiments in the thesis).
DEFAULT_R = 2.0      # attachment scale factor r in the attachment test
DEFAULT_TAU = 0.5    # attachment-fraction threshold (accept/reject component)
DEFAULT_BETA = 0.35  # internal/external scale factor in the rescue pass


def reachability_density(distance_matrix: np.ndarray,
                         knn_indices: np.ndarray,
                         k: int) -> np.ndarray:
    """Compute the reachability-based local density metric ``rho``.

    For every point *i* the average reachability distance towards its *k*
    nearest neighbours is computed and inverted:

        rho_i = k / (mean_j max(d_k(j), d(i, j)) + 1e-10)

    where ``d_k(j)`` is the k-th neighbour distance of neighbour ``j``.
    Using reachability distances (instead of raw distances) makes the
    metric more robust to local fluctuations than the plain k-th
    neighbour distance.

    Parameters
    ----------
    distance_matrix:
        Full ``n x n`` Euclidean distance matrix (zero diagonal).
    knn_indices:
        ``n x k`` matrix with the indices of the k nearest neighbours of
        every point (self excluded).
    k:
        Number of neighbours used in the metric.

    Returns
    -------
    np.ndarray of shape ``(n,)`` -- the density metric ``rho``.
    """
    n_points = distance_matrix.shape[0]
    rho = np.zeros(n_points)
    for i in range(n_points):
        neighbours = knn_indices[i, :]
        # k-th neighbour distance of every neighbour of i
        neighbour_k_dist = distance_matrix[neighbours, knn_indices[neighbours, k - 1]]
        # reachability distance from i to each neighbour
        reach_dist = np.maximum(neighbour_k_dist, distance_matrix[i, neighbours])
        avg_reach_dist = np.mean(reach_dist)
        rho[i] = k / (avg_reach_dist + 1e-10)
    return rho


def find_knee_point(values: np.ndarray) -> tuple:
    """Locate the knee (elbow) point of a descending curve.

    The knee is the point with the maximum perpendicular distance to the
    chord connecting the first and last points of the curve (a standard,
    parameter-free elbow detector).

    Parameters
    ----------
    values:
        Monotonically non-increasing 1-D array (e.g. sorted k-distances).

    Returns
    -------
    (knee_index, knee_value)
    """
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return 0, float(y[0]) if len(y) > 0 else 0.0

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        knee_index = len(y) // 2
    else:
        unit = line_vec / line_len
        point_vec = np.vstack((x, y)).T - p1
        cross = np.abs(point_vec[:, 0] * unit[1] - point_vec[:, 1] * unit[0])
        knee_index = int(np.argmax(cross))

    # avoid degenerate knees at the very end of the curve
    if knee_index > len(y) * 0.9:
        knee_index = int(len(y) * 0.75)
    return knee_index, float(y[knee_index])


def adaptive_dbscan(data_points: np.ndarray) -> tuple:
    """Run one *adaptive* DBSCAN invocation on a subset of the data.

    Both DBSCAN parameters are derived from the subset itself:

    * ``MinPts = max(3, floor(log2(n_s)), d + 1)`` (clamped to ``n_s - 1``)
    * ``eps``    = value of the knee point on the sorted MinPts-distance
      curve of the subset (falls back to the median k-distance when the
      knee is degenerate).

    Parameters
    ----------
    data_points:
        ``n_s x d`` array with the subset coordinates.

    Returns
    -------
    labels:
        DBSCAN labels for the subset (``-1`` = noise within the subset).
    eps:
        The radius that was used.
    min_samples:
        The MinPts value that was used.
    """
    n_points = data_points.shape[0]
    if n_points < 3:
        return np.zeros(n_points, dtype=int), 0.1, 3

    dim = data_points.shape[1] if data_points.ndim > 1 else 1
    min_samples = max(3, int(np.log2(n_points)), dim + 1)
    min_samples = min(min_samples, n_points - 1)

    distance_matrix = squareform(pdist(data_points, metric="euclidean"))

    # sorted k-distance curve (distance to the MinPts-th nearest neighbour)
    dist_for_knn = distance_matrix.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    sorted_indices = np.argsort(dist_for_knn, axis=1)
    k_idx = min(min_samples, sorted_indices.shape[1])
    k_distances = dist_for_knn[np.arange(n_points), sorted_indices[:, k_idx - 1]]

    sorted_k_distances = np.sort(k_distances)[::-1]
    _, optimal_eps = find_knee_point(sorted_k_distances)
    if optimal_eps <= 0 or np.isnan(optimal_eps):
        optimal_eps = float(np.median(k_distances))

    np.fill_diagonal(distance_matrix, 0.0)
    labels = DBSCAN(eps=optimal_eps, min_samples=min_samples,
                    metric="precomputed").fit_predict(distance_matrix)
    return labels, float(optimal_eps), int(min_samples)


def ar_mdbscan(data_points: np.ndarray,
               t: float,
               r: float = DEFAULT_R,
               tau: float = DEFAULT_TAU,
               beta: float = DEFAULT_BETA) -> np.ndarray:
    """Run the full AR-MDBSCAN pipeline.

    Parameters
    ----------
    data_points:
        ``n x d`` array with the input coordinates.
    t:
        Density threshold that separates low-density from high-density
        points.  This is the **only** user-tunable parameter of the method
        (required).
    r:
        Attachment scale factor of the attachment test (default ``2.0``).
    tau:
        Attachment-fraction threshold for accepting/rejecting a low-density
        component (default ``0.5``).
    beta:
        Internal/external scale factor used by the rescue pass
        (default ``0.35``).

    Returns
    -------
    np.ndarray of shape ``(n,)`` with integer cluster labels ``0..K-1``.
    The output is a complete partition: no point is left unlabelled.

    Raises
    ------
    ValueError
        If ``t`` is not provided.
    """
    if t is None:
        raise ValueError(
            "The density threshold 't' must be provided; it is the single "
            "user-tunable parameter of AR-MDBSCAN.")

    data_points = np.asarray(data_points, dtype=float)
    n_points = data_points.shape[0]
    if n_points == 0:
        return np.zeros(0, dtype=int)

    # ------------------------------------------------------------------
    # Step 1: global distance structure and data-driven neighbour count
    # ------------------------------------------------------------------
    k_neighbors = max(3, int(np.log2(n_points)))
    k_neighbors = min(k_neighbors, n_points - 1)

    distance_matrix = squareform(pdist(data_points, metric="euclidean"))
    dist_for_knn = distance_matrix.copy()
    np.fill_diagonal(dist_for_knn, np.inf)
    sorted_indices = np.argsort(dist_for_knn, axis=1)
    knn_indices = sorted_indices[:, :k_neighbors]
    # local scale of every point = distance to its k-th nearest neighbour
    k_dist_all = dist_for_knn[np.arange(n_points), sorted_indices[:, k_neighbors - 1]]

    # ------------------------------------------------------------------
    # Step 2: density metric, Step 3: split into low/high density sets
    # ------------------------------------------------------------------
    rho = reachability_density(distance_matrix, knn_indices, k_neighbors)
    low_density_indices = np.where(rho <= t)[0]
    high_density_indices = np.where(rho > t)[0]

    cluster_assignments = np.full(n_points, -1, dtype=int)
    next_label = 0

    # ------------------------------------------------------------------
    # Step 4 + 5: adaptive DBSCAN on L and the attachment test
    # ------------------------------------------------------------------
    noise_low_indices = np.array([], dtype=int)
    fringe_indices = np.array([], dtype=int)
    insignificant_indices = np.array([], dtype=int)

    if len(low_density_indices) > 0:
        subset_labels, _, _ = adaptive_dbscan(data_points[low_density_indices])

        clustered_mask = subset_labels != -1
        noise_low_indices = low_density_indices[~clustered_mask]

        for comp_id in np.unique(subset_labels[clustered_mask]):
            comp_points = low_density_indices[subset_labels == comp_id]

            # attachment fraction: share of points close (relative to the
            # anchor's local scale) to the nearest high-density anchor
            if len(high_density_indices) > 0:
                d_comp_hd = distance_matrix[np.ix_(comp_points, high_density_indices)]
                nearest_local = np.argmin(d_comp_hd, axis=1)
                d_to_anchor = d_comp_hd[np.arange(len(comp_points)), nearest_local]
                anchors = high_density_indices[nearest_local]
                attached_mask = d_to_anchor <= r * np.maximum(k_dist_all[anchors], 1e-12)
                attach_fraction = float(attached_mask.mean())
            else:
                attach_fraction = 0.0

            if attach_fraction >= tau:
                # mostly glued to the high-density region -> fringe/noise
                noise_low_indices = np.concatenate([noise_low_indices, comp_points])
            else:
                # independent low-density structure -> new cluster
                cluster_assignments[comp_points] = next_label
                next_label += 1

        # point-wise fringe test for the leftover low-density noise
        if len(noise_low_indices) > 0 and len(high_density_indices) > 0:
            d_noise_hd = distance_matrix[np.ix_(noise_low_indices, high_density_indices)]
            nearest_local = np.argmin(d_noise_hd, axis=1)
            d_hd = d_noise_hd[np.arange(len(noise_low_indices)), nearest_local]
            anchors = high_density_indices[nearest_local]
            attached_mask = d_hd <= r * np.maximum(k_dist_all[anchors], 1e-12)
            fringe_indices = noise_low_indices[attached_mask]
            insignificant_indices = noise_low_indices[~attached_mask]
        else:
            insignificant_indices = noise_low_indices.copy()

    # ------------------------------------------------------------------
    # Step 6a: adaptive DBSCAN on H united with the insignificant points
    # ------------------------------------------------------------------
    hd_noise_indices = np.array([], dtype=int)
    part_b_indices = np.concatenate(
        [high_density_indices, insignificant_indices]).astype(int)

    if len(part_b_indices) > 0:
        part_b_labels, _, _ = adaptive_dbscan(data_points[part_b_indices])
        valid_mask = part_b_labels != -1
        if np.any(valid_mask):
            labelled_points = part_b_indices[valid_mask]
            cluster_assignments[labelled_points] = part_b_labels[valid_mask] + next_label
            next_label += len(np.unique(part_b_labels[valid_mask]))
        hd_noise_indices = part_b_indices[~valid_mask]

    # ------------------------------------------------------------------
    # Step 6b: rescue pass over fringe points and high-density noise
    # ------------------------------------------------------------------
    rescue_pool = np.concatenate([fringe_indices, hd_noise_indices]).astype(int)
    unrescued = rescue_pool
    if len(rescue_pool) >= 2:
        rescue_labels, _, _ = adaptive_dbscan(data_points[rescue_pool])
        rescued_mask = np.zeros(len(rescue_pool), dtype=bool)
        already_labelled = np.where(cluster_assignments != -1)[0]

        for comp_id in np.unique(rescue_labels[rescue_labels != -1]):
            comp_global = rescue_pool[rescue_labels == comp_id]
            if len(comp_global) < 2:
                continue

            # internal scale: median nearest-neighbour distance inside component
            comp_dm = distance_matrix[np.ix_(comp_global, comp_global)].copy()
            np.fill_diagonal(comp_dm, np.inf)
            internal_scale = np.median(comp_dm.min(axis=1))

            # external scale: median distance to the closest labelled point
            if len(already_labelled) > 0:
                d_to_existing = distance_matrix[np.ix_(comp_global, already_labelled)]
                external_scale = np.median(d_to_existing.min(axis=1))
            else:
                external_scale = np.inf

            # genuine structures are internally tight relative to their gap
            is_genuine = (external_scale <= 0) or (internal_scale <= beta * external_scale)
            if is_genuine:
                cluster_assignments[comp_global] = next_label
                next_label += 1
                rescued_mask[rescue_labels == comp_id] = True

        unrescued = rescue_pool[~rescued_mask]

    # ------------------------------------------------------------------
    # Step 7: complete the partition
    # ------------------------------------------------------------------
    final_unlabelled = np.where(cluster_assignments == -1)[0]
    labelled = np.where(cluster_assignments != -1)[0]
    if len(final_unlabelled) > 0 and len(labelled) > 0:
        nearest_labelled = labelled[
            np.argmin(distance_matrix[np.ix_(final_unlabelled, labelled)], axis=1)]
        cluster_assignments[final_unlabelled] = cluster_assignments[nearest_labelled]

    # relabel consecutively from 0
    unique_labels = np.unique(cluster_assignments)
    return np.searchsorted(unique_labels, cluster_assignments)
