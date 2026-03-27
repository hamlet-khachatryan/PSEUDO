from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from analyse.water_sites.clustering import WaterSite
from analyse.water_sites.metrics import WaterSiteConsistency, SNR_NOISE_FLOOR
from analyse.water_sites.occupancy import SiteOccupancy
from analyse.water_sites.site_snr import PerStructureSiteSNR


# ---------------------------------------------------------------------------
# Nature-style figure defaults
# ---------------------------------------------------------------------------

_NATURE_RC: dict = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Single-column width in inches (Nature standard)
_W1 = 3.5
# Double-column width in inches
_W2 = 7.2

# ---------------------------------------------------------------------------
# Occupancy colour palette  (colorblind-safe categorical)
# ---------------------------------------------------------------------------

_OCC_PALETTE: Dict[str, str] = {
    "water":   "#2166AC",   # blue
    "ligand":  "#D6604D",   # red-orange
    "protein": "#4DAF4A",   # green
    "ion":     "#984EA3",   # purple
    "empty":   "#AAAAAA",   # neutral grey
}
_OCC_ORDER = ["water", "ligand", "protein", "ion", "empty"]

# Number of top sites annotated with site_id in each figure
_N_LABEL = 8

# Significance thresholds for the volcano plot
_P_THRESHOLD_STRICT = 0.01
_P_THRESHOLD_NOMINAL = 0.05


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_ax(ax: plt.Axes) -> None:
    """Remove top/right spines; keep ticks clean."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)


def _guide(ax: plt.Axes, x: Optional[float] = None, y: Optional[float] = None) -> None:
    """Draw subtle dashed reference line(s)."""
    kw = dict(color="#CCCCCC", linewidth=0.7, linestyle="--", zorder=0)
    if x is not None:
        ax.axvline(x, **kw)
    if y is not None:
        ax.axhline(y, **kw)


def _marker_area(n: np.ndarray, lo: float = 18.0, hi: float = 120.0) -> np.ndarray:
    """Map n_waters to marker area (matplotlib s parameter)."""
    mn, mx = float(n.min()), float(n.max())
    if mn == mx:
        return np.full_like(n, (lo + hi) / 2.0, dtype=float)
    return lo + (n - mn) / (mx - mn) * (hi - lo)


def _label_top(
    ax: plt.Axes,
    site_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    n: int = _N_LABEL,
) -> None:
    """Annotate the top-n finite points ranked by score."""
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(score)
    if not valid.any():
        return
    top = np.where(valid)[0][np.argsort(score[valid])[::-1][:n]]
    for i in top:
        ax.annotate(
            f"#{site_ids[i]}",
            xy=(x[i], y[i]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            color="#555555",
            clip_on=True,
        )


def _size_legend(ax: plt.Axes, n_waters: np.ndarray) -> None:
    """Add a discrete size legend for n_waters."""
    counts = sorted({int(n_waters.min()), int(np.median(n_waters)), int(n_waters.max())})
    handles = [
        plt.scatter([], [],
                    s=_marker_area(np.array([float(c)]))[0],
                    color="#888888", alpha=0.6,
                    edgecolors="#333333", linewidths=0.4,
                    label=str(c))
        for c in counts
    ]
    ax.legend(handles=handles, title="Waters\nmodelled", loc="lower right",
              handletextpad=0.4, labelspacing=0.3)


def _colorbar(fig, ax, sm, label: str) -> None:
    cb = fig.colorbar(sm, ax=ax, pad=0.03, shrink=0.85, aspect=18)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)


def _collect(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
) -> Tuple[np.ndarray, ...]:
    """Return parallel arrays: site_ids, snr_mean, consistency, water_freq, n_waters."""
    return (
        np.array([s.site_id for s in sites], dtype=np.int32),
        np.array([c.snr_mean_across_screen for c in consistencies], dtype=float),
        np.array([c.consistency_score for c in consistencies], dtype=float),
        np.array([c.water_frequency for c in consistencies], dtype=float),
        np.array([s.n_waters for s in sites], dtype=float),
    )


# ---------------------------------------------------------------------------
# Figure 1 — SNR vs consistency, coloured by water frequency
# ---------------------------------------------------------------------------

def _fig_snr_consistency(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    output_path: Path,
) -> None:
    """
    SNR vs consistency score coloured by water modelling frequency.

    Each point represents one conserved water site. The x-axis shows how
    strong the electron density signal is at the site (averaged over all
    structures), and the y-axis shows how uniformly that signal appears
    (consistency_score = 1 / (1 + CV); 1 = no variation between structures).
    Colour encodes the fraction of structures where a water is explicitly
    modelled at this site.

    Interpretation:
      Top-right  (SNR > 1, consistency > 0.5): structurally conserved,
          high-confidence waters — stable across conditions.
      Bottom-right (SNR > 1, consistency < 0.5): strong signal but variable —
          the site may be displaced by ligands in some experiments.
      Top-left   (SNR < 1, consistency > 0.5): consistently present but weak —
          surface / partially disordered waters.
      Bottom-left: likely noise.
    """
    site_ids, snr, cons, freq, n_w = _collect(sites, consistencies)

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_W1, _W1 * 0.88))

        valid = np.isfinite(snr) & np.isfinite(cons)
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap("plasma")

        sc = ax.scatter(
            snr[valid], cons[valid],
            c=freq[valid], cmap=cmap, norm=norm,
            s=_marker_area(n_w)[valid],
            alpha=0.80, linewidths=0.35, edgecolors="#333333", zorder=3,
        )

        _guide(ax, x=SNR_NOISE_FLOOR, y=0.5)
        _clean_ax(ax)

        ax.set_xlabel("Mean SNR across screen  (site-radius sphere)")
        ax.set_ylabel("Consistency score  [1 / (1 + CV)]")
        ax.set_title("SNR vs. signal consistency")
        ax.set_ylim(bottom=-0.04)

        # Soft quadrant labels
        xl, xr = ax.get_xlim()
        yl, yr = ax.get_ylim()
        xm = (SNR_NOISE_FLOOR + xr) * 0.54
        _kw = dict(fontsize=5.5, color="#AAAAAA", ha="center", va="center",
                   style="italic", zorder=0)
        ax.text(xm, (0.5 + yr) * 0.52, "essential\nwaters", **_kw)
        ax.text(xm, (yl + 0.5) * 0.52, "displaced\nsites", **_kw)
        ax.text((xl + SNR_NOISE_FLOOR) * 0.5, (0.5 + yr) * 0.52, "ghost\nwaters", **_kw)

        _colorbar(fig, ax, ScalarMappable(norm=norm, cmap=cmap), "Water frequency")
        _label_top(ax, site_ids, snr, cons, snr * cons)
        _size_legend(ax, n_w)

        fig.savefig(output_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — SNR vs consistency, coloured by most-common occupancy type
# ---------------------------------------------------------------------------

def _fig_snr_by_occupancy(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    output_path: Path,
) -> None:
    """
    SNR vs consistency score with points coloured by the most common occupancy
    type observed at each site across the screen.

    This overlay on the SNR-consistency space reveals which sites are
    predominantly occupied by water, protein atoms, ligands, or ions — and
    whether that occupancy correlates with signal strength and consistency.
    Sites coloured orange (ligand) that sit in the high-SNR / low-consistency
    quadrant are prime candidates for water-displacing fragment events.
    """
    site_ids, snr, cons, freq, n_w = _collect(sites, consistencies)
    occ_labels = [c.most_common_occupancy for c in consistencies]

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_W1, _W1 * 0.88))

        valid = np.isfinite(snr) & np.isfinite(cons)
        sizes = _marker_area(n_w)

        for occ in _OCC_ORDER:
            mask = valid & np.array([lbl == occ for lbl in occ_labels])
            if not mask.any():
                continue
            ax.scatter(
                snr[mask], cons[mask],
                color=_OCC_PALETTE[occ],
                s=sizes[mask],
                alpha=0.80, linewidths=0.35, edgecolors="#333333",
                zorder=3, label=occ,
            )

        _guide(ax, x=SNR_NOISE_FLOOR, y=0.5)
        _clean_ax(ax)

        ax.set_xlabel("Mean SNR across screen  (site-radius sphere)")
        ax.set_ylabel("Consistency score  [1 / (1 + CV)]")
        ax.set_title("SNR vs. consistency by occupancy type")
        ax.set_ylim(bottom=-0.04)

        ax.legend(
            title="Most common\noccupancy",
            loc="upper left",
            markerscale=0.9,
            handletextpad=0.3,
            labelspacing=0.25,
        )

        _label_top(ax, site_ids, snr, cons, snr * cons)

        fig.savefig(output_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Volcano plot: effect size vs. statistical significance
# ---------------------------------------------------------------------------

def _fig_volcano(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    output_path: Path,
) -> None:
    """
    Volcano plot of SNR effect size vs. statistical significance.

    X-axis: mean SNR − SNR_NOISE_FLOOR (deviation above/below the noise floor;
        positive = above background, negative = below).
    Y-axis: −log₁₀(p-value) from a one-sample t-test
        (H₀: μ_SNR ≤ 1.0, Hₐ: μ_SNR > 1.0).
    Colour: consistency score.

    Dashed lines mark the nominal (p = 0.05) and strict (p = 0.01)
    significance thresholds. Points require ≥ 3 per-structure observations
    to have a valid p-value; sites with fewer observations are plotted on the
    x-axis at y = 0.

    The top-right corner identifies sites with both large effect size AND
    high statistical confidence — the most trustworthy water sites in the
    screen. These are the sites worth examining in electron density maps.
    """
    site_ids, snr, cons, freq, n_w = _collect(sites, consistencies)
    pvalues = np.array([c.snr_pvalue for c in consistencies], dtype=float)

    effect = snr - SNR_NOISE_FLOOR
    with np.errstate(divide="ignore"):
        neg_log_p = -np.log10(np.where(pvalues > 0, pvalues, np.nan))

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_W1, _W1 * 0.88))

        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap("viridis")
        sizes = _marker_area(n_w)

        # Sites with valid p-values
        valid_p = np.isfinite(effect) & np.isfinite(neg_log_p)
        if valid_p.any():
            ax.scatter(
                effect[valid_p], neg_log_p[valid_p],
                c=cons[valid_p], cmap=cmap, norm=norm,
                s=sizes[valid_p],
                alpha=0.80, linewidths=0.35, edgecolors="#333333", zorder=3,
            )

        # Sites without enough observations (p = NaN): plot at y = 0 in grey
        no_p = np.isfinite(effect) & ~np.isfinite(neg_log_p)
        if no_p.any():
            ax.scatter(
                effect[no_p], np.zeros(no_p.sum()),
                color="#CCCCCC", s=sizes[no_p] * 0.6,
                alpha=0.6, linewidths=0.3, edgecolors="#999999",
                zorder=2, label="n < 3 (no test)",
            )
            ax.legend(loc="upper left", handletextpad=0.3)

        # Reference lines
        _guide(ax, x=0)
        for thresh, label in [
            (_P_THRESHOLD_NOMINAL, f"p = {_P_THRESHOLD_NOMINAL}"),
            (_P_THRESHOLD_STRICT, f"p = {_P_THRESHOLD_STRICT}"),
        ]:
            y_val = -math.log10(thresh)
            ax.axhline(y_val, color="#CCCCCC", linewidth=0.7, linestyle="--", zorder=0)
            ax.text(
                ax.get_xlim()[1] * 0.98, y_val + 0.05, label,
                ha="right", va="bottom", fontsize=5.5, color="#AAAAAA",
            )

        _clean_ax(ax)
        ax.set_xlabel(f"Mean SNR − {SNR_NOISE_FLOOR:.1f}  (effect size)")
        ax.set_ylabel("−log₁₀(p-value)  [one-sample t-test vs. noise floor]")
        ax.set_title("Volcano plot — SNR significance")

        _colorbar(fig, ax, ScalarMappable(norm=norm, cmap=cmap), "Consistency score")

        composite = np.where(valid_p, effect * neg_log_p, -np.inf)
        _label_top(ax, site_ids, effect, neg_log_p, composite)

        fig.savefig(output_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Fragment correlation: SNR when water vs. when ligand occupies
# ---------------------------------------------------------------------------

def _snr_by_occupancy_group(
    sites: List[WaterSite],
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
) -> Dict[int, Dict[str, List[float]]]:
    """
    For each site, bucket per-structure mean-SNR values by occupancy type.

    Returns {site_id: {occ_type_value: [snr_mean, ...]}}.
    Only records with a finite SNR mean are included.
    """
    occ_index: Dict[Tuple[int, str], str] = {
        (o.site_id, o.stem): o.occupancy_type.value
        for o in per_structure_occ
    }
    groups: Dict[int, Dict[str, List[float]]] = {
        s.site_id: defaultdict(list) for s in sites
    }
    for rec in per_structure_snr:
        occ = occ_index.get((rec.site_id, rec.stem), "empty")
        val = rec.snr_site_radius.mean if rec.snr_site_radius is not None else None
        if val is not None and math.isfinite(val):
            groups[rec.site_id][occ].append(val)
    return groups


def _fig_fragment_correlation(
    sites: List[WaterSite],
    consistencies: List[WaterSiteConsistency],
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
    output_path: Path,
    min_group: int = 2,
) -> None:
    """
    Fragment correlation scatter: mean SNR at the water site centroid when
    a water is present vs. when a ligand occupies that position.

    Each point represents one site where both a "water group" (structures
    where the site is occupied by a modelled water) and a "ligand group"
    (structures where a ligand occupies the same position) exist with at
    least min_group observations each.

    Interpretation:
      Points on the diagonal: ligand maintains the same density signal at
          the water centroid as the water itself — the ligand passes through
          or closely overlaps the water position.
      Points above the diagonal (SNR_water > SNR_ligand): water gives
          stronger focal density than the ligand at this exact location —
          the ligand may be positioned nearby but not centred here.
      Points below the diagonal: ligand gives stronger density at this point
          than water did — suggestive of a heavy atom in the fragment being
          positioned right at the water centroid.
      Size: number of ligand structures contributing to the ligand-group mean.
      Colour: water_frequency.
    """
    site_ids_all, _, _, freq_all, n_w_all = _collect(sites, consistencies)
    groups = _snr_by_occupancy_group(sites, per_structure_snr, per_structure_occ)

    rows = []  # (site_id, snr_water, snr_ligand, freq, n_ligand)
    for site, freq in zip(sites, [c.water_frequency for c in consistencies]):
        g = groups.get(site.site_id, {})
        w_vals = g.get("water", [])
        l_vals = g.get("ligand", [])
        if len(w_vals) >= min_group and len(l_vals) >= min_group:
            rows.append((
                site.site_id,
                float(np.mean(w_vals)),
                float(np.mean(l_vals)),
                freq,
                len(l_vals),
            ))

    with plt.rc_context(_NATURE_RC):
        fig, ax = plt.subplots(figsize=(_W1, _W1 * 0.88))

        if not rows:
            ax.text(0.5, 0.5,
                    f"No sites with ≥ {min_group} structures\nin both water and ligand groups",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="#888888")
            _clean_ax(ax)
            ax.set_title("Fragment correlation")
            fig.savefig(output_path)
            plt.close(fig)
            return

        sids  = np.array([r[0] for r in rows], dtype=np.int32)
        x_w   = np.array([r[1] for r in rows], dtype=float)   # SNR water
        y_l   = np.array([r[2] for r in rows], dtype=float)   # SNR ligand
        freqs = np.array([r[3] for r in rows], dtype=float)
        n_lig = np.array([r[4] for r in rows], dtype=float)

        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.get_cmap("plasma")
        sizes = _marker_area(n_lig, lo=20.0, hi=140.0)

        sc = ax.scatter(
            x_w, y_l,
            c=freqs, cmap=cmap, norm=norm,
            s=sizes,
            alpha=0.80, linewidths=0.35, edgecolors="#333333", zorder=3,
        )

        # Diagonal reference line (equal SNR)
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, color="#CCCCCC", linewidth=0.8, linestyle="--",
                zorder=0, label="equal SNR")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Guide lines at noise floor
        _guide(ax, x=SNR_NOISE_FLOOR, y=SNR_NOISE_FLOOR)

        _clean_ax(ax)
        ax.set_xlabel("Mean SNR at site  |  water-occupied structures")
        ax.set_ylabel("Mean SNR at site  |  ligand-occupied structures")
        ax.set_title("Fragment correlation")
        ax.set_aspect("equal", adjustable="box")

        _colorbar(fig, ax, ScalarMappable(norm=norm, cmap=cmap), "Water frequency")

        # Above/below diagonal labels
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.text(xl[0] * 0.6 + xl[1] * 0.4, yl[0] * 0.1 + yl[1] * 0.9,
                "water > ligand signal", fontsize=5.5, color="#AAAAAA",
                style="italic", ha="center")
        ax.text(xl[0] * 0.1 + xl[1] * 0.6, yl[0] * 0.9 + yl[1] * 0.1,
                "ligand > water signal", fontsize=5.5, color="#AAAAAA",
                style="italic", ha="center")

        # Size legend for n_ligand structures
        counts = sorted({int(n_lig.min()), int(np.median(n_lig)), int(n_lig.max())})
        handles = [
            plt.scatter([], [],
                        s=_marker_area(np.array([float(c)], dtype=float), 20, 140)[0],
                        color="#888888", alpha=0.6,
                        edgecolors="#333333", linewidths=0.4,
                        label=str(c))
            for c in counts
        ]
        ax.legend(handles=handles, title="Ligand\nstructures",
                  loc="upper left", handletextpad=0.4, labelspacing=0.3)

        _label_top(ax, sids, x_w, y_l, np.abs(x_w - y_l))

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
    Produce all four diagnostic figures and write them to figures_dir.

    Figures produced:
        01_snr_vs_consistency.png   — SNR vs. consistency, coloured by water frequency
        02_snr_by_occupancy.png     — same axes, coloured by most-common occupancy type
        03_volcano.png              — effect size vs. −log₁₀(p-value) from t-test
        04_fragment_correlation.png — SNR when water present vs. when ligand present

    Args:
        sites: WaterSite list from cluster_water_observations.
        consistencies: Matching WaterSiteConsistency list from compute_consistency.
        per_structure_snr: Flat list of PerStructureSiteSNR from all structures.
        per_structure_occ: Flat list of SiteOccupancy from all structures.
        figures_dir: Directory where PNGs are written (created if absent).
    """
    if not sites:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    _fig_snr_consistency(
        sites, consistencies,
        figures_dir / "01_snr_vs_consistency.png",
    )
    _fig_snr_by_occupancy(
        sites, consistencies,
        figures_dir / "02_snr_by_occupancy.png",
    )
    _fig_volcano(
        sites, consistencies,
        figures_dir / "03_volcano.png",
    )
    _fig_fragment_correlation(
        sites, consistencies,
        per_structure_snr, per_structure_occ,
        figures_dir / "04_fragment_correlation.png",
    )
