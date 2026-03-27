from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from analyse.water_sites.clustering import WaterSite
from analyse.water_sites.metrics import WaterSiteConsistency, SNR_NOISE_FLOOR
from analyse.water_sites.occupancy import SiteOccupancy
from analyse.water_sites.site_snr import PerStructureSiteSNR


# ---------------------------------------------------------------------------
# Nature-style rcParams
# ---------------------------------------------------------------------------

_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
}

_FIG_W = 4.0   # inches — comfortable single column
_FIG_H = 3.8

_POINT_COLOR = "#2166AC"   # single-color version
_EDGE_COLOR  = "#1a1a1a"
_GUIDE_COLOR = "#CCCCCC"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _open_ax(ax: plt.Axes) -> None:
    """Remove top/right spines; keep left and bottom only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


def _noise_line(ax: plt.Axes) -> None:
    """Subtle horizontal guide at the SNR noise floor."""
    ax.axhline(
        SNR_NOISE_FLOOR,
        color=_GUIDE_COLOR, linewidth=0.8, linestyle="--", zorder=0,
    )


def _label_top(
    ax: plt.Axes,
    site_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    n: int = 8,
) -> None:
    """Annotate the top-n sites ranked by score (finite values only)."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(score)
    if not valid.any():
        return
    top = np.where(valid)[0][np.argsort(score[valid])[::-1][:n]]
    for i in top:
        ax.annotate(
            f"#{site_ids[i]}",
            xy=(x[i], y[i]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6,
            color="#555555",
            clip_on=True,
        )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _arrays(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (site_ids, snr_mean, pairwise_diff, water_freq)."""
    return (
        np.array([s.site_id for s in sites], dtype=np.int32),
        np.array([c.snr_mean_across_screen for c in consistencies], dtype=float),
        np.array([c.mean_pairwise_snr_diff for c in consistencies], dtype=float),
        np.array([c.water_frequency for c in consistencies], dtype=float),
    )


# ---------------------------------------------------------------------------
# Figure A — plain scatter  (no colour / size encoding)
# ---------------------------------------------------------------------------

def _fig_plain(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    output_path: Path,
) -> None:
    """
    Mean SNR (y) vs. mean pairwise |ΔSNR| (x) — one dot per water site.

    X — mean pairwise absolute difference between per-structure SNR means
        [SNR units].  Lower = all structures produce nearly the same SNR
        at this site (high reproducibility).
    Y — mean SNR across all structures at the site-radius sphere [SNR units].

    The most structurally important sites cluster towards the top-left:
    high signal, low inter-structure variability.  The dashed line marks
    the SNR noise floor (1.0); sites below it are not reliably above
    background in any structure.
    """
    ids, snr, diff, _ = _arrays(sites, consistencies)
    valid = np.isfinite(snr) & np.isfinite(diff)

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

        ax.scatter(
            diff[valid], snr[valid],
            color=_POINT_COLOR,
            s=20,
            alpha=0.55,
            linewidths=0.3,
            edgecolors=_EDGE_COLOR,
            zorder=3,
            rasterized=len(sites) > 300,
        )

        _noise_line(ax)
        _open_ax(ax)

        ax.set_xlabel("Mean pairwise |ΔSNR|  (inter-structure variability)")
        ax.set_ylabel("Mean SNR")
        ax.set_title("Water site SNR vs. reproducibility")

        # Rank: high SNR, low variability
        score = np.where(valid, snr / (diff + 1e-9), -np.inf)
        _label_top(ax, ids, diff, snr, score)

        fig.savefig(output_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B — coloured by water frequency
# ---------------------------------------------------------------------------

def _fig_colored(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    output_path: Path,
) -> None:
    """
    Same axes as _fig_plain; points are coloured by water_frequency —
    the fraction of structures where a water was explicitly modelled at
    this site.  High frequency (yellow) sites that also sit in the top-left
    (high SNR, low variability) are the most consistently modelled and
    reliably detectable waters in the screen.
    """
    ids, snr, diff, freq = _arrays(sites, consistencies)
    valid = np.isfinite(snr) & np.isfinite(diff)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("plasma")

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(_FIG_W + 0.6, _FIG_H))

        sc = ax.scatter(
            diff[valid], snr[valid],
            c=freq[valid],
            cmap=cmap, norm=norm,
            s=20,
            alpha=0.70,
            linewidths=0.3,
            edgecolors=_EDGE_COLOR,
            zorder=3,
            rasterized=len(sites) > 300,
        )

        _noise_line(ax)
        _open_ax(ax)

        ax.set_xlabel("Mean pairwise |ΔSNR|  (inter-structure variability)")
        ax.set_ylabel("Mean SNR")
        ax.set_title("Water site SNR vs. reproducibility")

        cb = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, pad=0.03, shrink=0.85, aspect=20,
        )
        cb.set_label("Water frequency", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.5)

        score = np.where(valid, snr / (diff + 1e-9), -np.inf)
        _label_top(ax, ids, diff, snr, score)

        fig.savefig(output_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_water_analysis_figures(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
    figures_dir: Path,
) -> None:
    """
    Write diagnostic figures to figures_dir.

    Figures produced
    ----------------
    01_snr_vs_reproducibility.png
        Mean SNR (y) vs. mean pairwise |ΔSNR| (x).  Plain uniform markers.
        Each point = one conserved water site.

    02_snr_vs_reproducibility_freq.png
        Same axes; points coloured by the fraction of structures where a
        water was explicitly modelled at the site (water frequency).

    Args:
        sites: WaterSite list from cluster_water_observations.
        consistencies: Matching WaterSiteConsistency list.
        per_structure_snr: Flat list of PerStructureSiteSNR (unused here,
            kept for API consistency with pipeline).
        per_structure_occ: Flat list of SiteOccupancy (unused here).
        figures_dir: Directory where PNGs are written (created if absent).
    """
    if not sites:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    _fig_plain(
        sites, consistencies,
        figures_dir / "01_snr_vs_reproducibility.png",
    )
    _fig_colored(
        sites, consistencies,
        figures_dir / "02_snr_vs_reproducibility_freq.png",
    )
