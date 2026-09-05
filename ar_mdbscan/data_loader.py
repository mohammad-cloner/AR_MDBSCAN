# -*- coding: utf-8 -*-
"""Dataset loading utilities for AR-MDBSCAN.

Supported formats
-----------------
* ``.mat``  -- MATLAB files containing the keys ``data``/``X`` and
  ``label``/``y``/``target`` (any combination).
* ``.txt`` / ``.dat`` / ``.tsv`` / ``.csv`` / ``.data`` / ``.arff`` --
  plain numeric tables.  Comment lines (``%``, ``#``, ``//``), ARFF
  headers and numeric headers (``n d``) are skipped automatically and the
  separator (space / tab / comma / semicolon) is detected automatically.
  The label column is auto-detected: the *last* column is treated as a
  label when it contains integers forming a reasonable number of classes
  (2 .. min(50, n/5)).

Returns ``(X, y)`` where ``y`` is ``None`` when no label column is found.
"""

from __future__ import annotations

import os
import numpy as np

TEXT_EXTS = (".txt", ".dat", ".tsv", ".csv", ".data", ".arff")


def _parse_text_rows(path: str) -> np.ndarray:
    """Read a numeric table, dropping comments and non-numeric lines."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("%", "#", "//")):
                continue
            if stripped[0] == "@":          # ARFF header
                continue
            tokens = (stripped
                      .replace("\t", " ")
                      .replace(",", " ")
                      .replace(";", " ")
                      .split())
            try:
                rows.append([float(tok) for tok in tokens])
            except ValueError:
                continue                    # e.g. a column-name line
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    return np.array(rows, dtype=float)


def _drop_numeric_header(arr: np.ndarray) -> np.ndarray:
    """Remove a leading ``n d`` header line when present (e.g. t4.8k.txt)."""
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
        return arr
    first = arr[0]
    if not (np.all(np.isfinite(first)) and np.all(first == np.round(first))):
        return arr
    if first[0] == arr[1:].shape[0] or first[1] == arr[1:].shape[1]:
        return arr[1:]
    return arr


def _detect_label_column(arr: np.ndarray) -> bool:
    """True when the *last* column looks like an integer class label."""
    if arr.ndim != 2 or arr.shape[1] < 2:
        return False
    last = arr[:, -1]
    if not np.all(np.isfinite(last)) or not np.all(last == np.round(last)):
        return False
    n_classes = len(np.unique(last))
    n = arr.shape[0]
    return 2 <= n_classes <= max(2, min(50, n // 5))


def load_dataset(path: str) -> tuple:
    """Load a dataset and return ``(X, y)``.

    ``y`` is ``None`` when no label column could be detected.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(path)
        X = y = None
        for key in ("data", "Data", "X", "x"):
            if key in mat:
                X = np.asarray(mat[key], dtype=float)
                break
        for key in ("label", "Label", "labels", "y", "Y", "target", "class", "gt"):
            if key in mat:
                y = np.asarray(mat[key]).ravel()
                break
        if X is None:
            raise ValueError(f"no 'data'/'X' key found in {path}")
        if y is not None and len(y) != len(X):
            y = None
        return X, y

    if ext in TEXT_EXTS:
        arr = _parse_text_rows(path)
        arr = _drop_numeric_header(arr)
        if _detect_label_column(arr):
            return arr[:, :-1], arr[:, -1]
        return arr, None

    raise ValueError(f"unsupported file extension: {ext}")
