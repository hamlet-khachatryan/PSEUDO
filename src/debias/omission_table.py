from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Dict
import numpy as np
import csv

from src.debias.omission_sampler import AnyID


def _id_to_str(item: AnyID) -> str:
    """Canonical string representation for an ID tuple."""
    return "|".join(map(str, item))


def _str_to_id(s: str) -> Tuple[str, ...]:
    """Reverse of _id_to_str (returns tuple of strings)."""
    return tuple(s.split("|"))


def build_omission_matrix(
    all_ids: Sequence[AnyID], selections: Sequence[Sequence[AnyID]]
) -> Tuple[List[str], np.ndarray]:
    """
    Build a boolean omission matrix.

    Args:
        all_ids: ordered sequence of every item ID (from extract_ids).
        selections: list of omitted selections to omit.

    Returns:
        (id_strings, matrix) where:
          - id_strings is a list[str] stable ID strings in the same order as all_ids
          - matrix is a numpy bool array of shape (n_items, n_samples)
            matrix[i, j] == True means item i was omitted in sample j.

    Complexity:
        - building a lookup set per sample and vectorized marking -> O(n_items * n_samples)
    """
    n_items = len(all_ids)
    n_samples = len(selections)
    id_strings = [_id_to_str(x) for x in all_ids]

    idx_map: Dict[str, int] = {s: i for i, s in enumerate(id_strings)}
    mat = np.zeros((n_items, n_samples), dtype=bool)

    for j, sel in enumerate(selections):
        if not sel:
            continue
        indices = [idx_map[s] for s in (_id_to_str(x) for x in sel) if s in idx_map]
        if len(indices) == 0:
            continue
        mat[indices, j] = True

    return id_strings, mat


def omission_sparse_map(id_strings: Sequence[str], matrix: np.ndarray) -> Dict[str, List[int]]:
    """
    Convert omission matrix to a sparse mapping id -> list of sample indices.

    Args:
        id_strings: list[str] of the item ids (row order).
        matrix: boolean numpy array shape (n_items, n_samples)

    Returns:
        dict mapping id_string -> sorted list of sample indices where the id was omitted.
    """
    n_items, n_samples = matrix.shape
    out: Dict[str, List[int]] = {}
    if n_items == 0 or n_samples == 0:
        for i, sid in enumerate(id_strings):
            out[sid] = []
        return out

    rows, cols = np.nonzero(matrix)

    for r, c in zip(rows, cols):
        sid = id_strings[r]
        out.setdefault(sid, []).append(int(c))

    for sid in id_strings:
        out.setdefault(sid, [])
    return out

def save_omission_csv(
    out_path: Path | str, id_strings: Sequence[str], matrix: np.ndarray, sample_prefix: str = "sample_"
) -> None:
    """
    Save omission matrix as CSV: first column 'id', then one column per sample (0/1).

    Args:
        out_path: Path where to write CSV. Uses pathlib.Path.
        id_strings: list of id strings (row order).
        matrix: boolean matrix (n_items, n_samples).
        sample_prefix: prefix for sample columns.
    """
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_items, n_samples = matrix.shape

    header = ["id"] + [f"{sample_prefix}{i}" for i in range(n_samples)]

    with outp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, sid in enumerate(id_strings):
            row = [sid] + [int(matrix[i, j]) for j in range(n_samples)]
            writer.writerow(row)
