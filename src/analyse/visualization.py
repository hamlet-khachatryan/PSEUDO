from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from analyse.muse.scoring import MUSEResult

ResidueDict = Dict[int, float]

_WATER_NAMES: frozenset = frozenset({"HOH", "WAT", "DOD", "H2O", "SOL"})

_SCORE_ATTR = {
    "musem": "musem_score",
    "min": "min_atom_score",
    "median": "median_atom_score",
    "max": "max_atom_score",
}

def edia_colormap() -> mcolors.LinearSegmentedColormap:
    """Red → pink → blue colormap for quality scale (0 – zeta)"""
    colours = [
        (0.0, "#D62728"),
        (0.4 / 1.2, "#FF9896"),
        (0.8 / 1.2, "#AEC7E8"),
        (1.0, "#1F77B4"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("edia", colours)


def pvalue_colormap() -> mcolors.LinearSegmentedColormap:
    """Red → yellow → green colormap for p-value maps [0, 1]"""
    colours = [
        (0.0, "#D62728"),
        (0.5, "#FFDD71"),
        (1.0, "#2CA02C"),
    ]
    return mcolors.LinearSegmentedColormap.from_list("pvalue", colours)


def extract_residue_scores(
    result: MUSEResult,
    score_field: str = "musem",
    chain_id: Optional[str] = None,
) -> ResidueDict:
    """
    Extract per-residue scores from a MUSEResult into a {seq_id: value} dict.

    Args:
        result: Completed MUSEResult.
        score_field: One of 'musem' (default), 'min', 'median', 'max'.
        chain_id: If given, restrict to residues on this chain.
    """
    if score_field not in _SCORE_ATTR:
        raise ValueError(f"score_field must be one of {list(_SCORE_ATTR)}, got '{score_field}'.")
    attr = _SCORE_ATTR[score_field]
    return {
        r.residue_seq_id: getattr(r, attr)
        for r in result.residue_scores
        if chain_id is None or r.chain_id == chain_id
    }

def plot_residue_profile(
    scores: ResidueDict,
    title: str = "MUSE Residue Profile",
    figsize: Tuple[float, float] = (14, 4),
    edia_thresholds: bool = True,
) -> Figure:
    """
    Line plot of per-residue MUSE scores.

    Args:
        scores: {seq_id: score} dict (from extract_residue_scores())
        edia_thresholds: If True, draw dashed lines at 0.4 and 0.8
    """
    seqs = sorted(scores)
    vals = np.array([scores[s] for s in seqs], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(seqs, vals, alpha=0.25, color="#4878CF")
    ax.plot(seqs, vals, color="#4878CF", linewidth=1.4)

    if edia_thresholds:
        for y, color in [(0.4, "#D65F5F"), (0.8, "#E8A838")]:
            ax.axhline(y, color=color, linewidth=0.9, linestyle="--", alpha=0.8, label=str(y))
        ax.legend(fontsize=9, loc="upper right")

    ax.set_xlabel("Residue sequence")
    ax.set_ylabel("MUSE score")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_water_support(
    result: MUSEResult,
    threshold: float = 0.5,
    chain_id: Optional[str] = None,
    score_field: str = "musem",
    title: str = "Water Density Support",
    figsize: Tuple[float, float] = (10, 5),
) -> Figure:
    """
    Rank plot of per-water MUSE scores with a removal threshold line

    Args:
        result: Completed MUSEResult
        threshold: Waters below this score are coloured red
        chain_id: If given, restrict to this chain
        score_field: One of 'musem' (default), 'min', 'median', 'max'
    """
    if score_field not in _SCORE_ATTR:
        raise ValueError(f"score_field must be one of {list(_SCORE_ATTR)}, got '{score_field}'.")
    attr = _SCORE_ATTR[score_field]

    raw = [
        getattr(r, attr)
        for r in result.residue_scores
        if r.residue_name.upper() in _WATER_NAMES
        and (chain_id is None or r.chain_id == chain_id)
    ]
    if not raw:
        raise ValueError("No water residues found in result.")

    scores = np.sort(raw)[::-1]
    ranks = np.arange(1, len(scores) + 1)
    colors = ["#2CA02C" if s >= threshold else "#D62728" for s in scores]

    n_keep = int((scores >= threshold).sum())
    n_remove = len(scores) - n_keep

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ranks, scores, c=colors, s=18, alpha=0.85, linewidths=0)
    ax.axhline(threshold, color="#555555", linewidth=1.4, linestyle="--",
               label=f"Threshold = {threshold:.2f}  (keep {n_keep}, remove {n_remove})")
    ax.set_xlabel("Rank (highest first)")
    ax.set_ylabel(f"Score ({score_field})")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
