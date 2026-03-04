from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from muse.config import (
    AggregationConfig,
    DensityScoreConfig,
    MapNormalizationConfig,
    MUSEConfig,
)
from muse.scoring import AtomScore, MUSEResult, ResidueScore


ResidueDict = Dict[int, float]


def electron_density_config() -> MUSEConfig:
    """
    Return a MUSEConfig for standard 2Fo-Fc electron density maps.
    Returns:
        MUSEConfig with normalize=True, zeta=1.2
    """
    return MUSEConfig(
        density_score=DensityScoreConfig(zeta=1.2, use_truncation=True),
        map_normalization=MapNormalizationConfig(normalize=True),
    )

def snr_map_config(zeta: float = 5.0) -> MUSEConfig:
    """
    Return a MUSEConfig  for SNR CCP4 maps.

    SNR values are not bounded by construction, raw values are used without
    z-score normalisation.  The upper truncation threshold *zeta* is set to a
    higher default (5.0) compared to electron density maps to avoid clipping
    meaningful high-SNR regions
    Args:
        zeta: Upper truncation threshold
    Returns:
        MUSEConfig
    """
    return MUSEConfig(
        density_score=DensityScoreConfig(zeta=zeta, use_truncation=True),
        map_normalization=MapNormalizationConfig(normalize=False),
    )


def pvalue_map_config() -> MUSEConfig:
    """
    Return a MUSEConfig  for p-value CCP4 maps (range [0, 1])

    P-value maps encode significance probabilities, raw values are used
    without z-score normalisation

    Returns:
        MUSEConfig
    """
    return MUSEConfig(
        density_score=DensityScoreConfig(zeta=1.0, use_truncation=True),
        map_normalization=MapNormalizationConfig(normalize=False),
    )

def apply_transform(
    values: np.ndarray,
    method: str = "none",
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply a display transform to an array of per-residue scores

    Args:
        values: 1-D float array of scores
        method: Transform name.  One of:

            * ``'none'`` — identity
            * ``'log1p'`` — ``log(1 + scale * x)``
            * ``'sqrt'`` — ``sqrt(scale * max(x, 0))``
            * ``'tanh'`` — ``tanh(x / scale)``, mapping to [0, 1).  Use *scale* ≈ mean expected value for a good spread
            * ``'sigmoid'`` — ``1 / (1 + exp(-x / scale))``, mapping ℝ → (0,1)
            * ``'neg_log10'`` — ``-log10(max(x, 1e-10))``.  Converts p-values to a significance scale
            * ``'clip_norm'`` — clip at *scale* then divide by *scale* to normalise to [0, 1]

        scale: Scale / range parameter whose meaning depends on *method*

    Returns:
        Transformed float array
    Raises:
        ValueError: If *method* is not recognised
    """
    x = np.asarray(values, dtype=float)
    if method == "none":
        return x
    elif method == "log1p":
        return np.log1p(np.maximum(x * scale, 0.0))
    elif method == "sqrt":
        return np.sqrt(np.maximum(x * scale, 0.0))
    elif method == "tanh":
        return np.tanh(x / max(scale, 1e-12))
    elif method == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x / max(scale, 1e-12)))
    elif method == "neg_log10":
        return -np.log10(np.maximum(x, 1e-10))
    elif method == "clip_norm":
        s = max(scale, 1e-12)
        return np.clip(x, 0.0, s) / s
    else:
        raise ValueError(
            f"Unknown transform '{method}'. Choose from: none, log1p, sqrt, "
            "tanh, sigmoid, neg_log10, clip_norm."
        )

def extract_residue_scores(
    result: MUSEResult,
    score_field: str = "musem",
    chain_id: Optional[str] = None,
) -> ResidueDict:
    """
    Extract per-residue scores from a MUSEResult into a plain dict

    Args:
        result: Completed MUSEResult
        score_field: Which score to extract

            * ``'musem'`` (default) — MUSEm power-mean aggregate
            * ``'min'`` — minimum per-atom score in the residue
            * ``'median'`` — median per-atom score
            * ``'max'`` — maximum per-atom score

        chain_id: If given restrict to residues on this chain

    Returns:
        Dict mapping residue sequence number to score value
    """
    data: ResidueDict = {}
    for r in result.residue_scores:
        if chain_id is not None and r.chain_id != chain_id:
            continue
        if score_field == "musem":
            data[r.residue_seq_id] = r.musem_score
        elif score_field == "min":
            data[r.residue_seq_id] = r.min_atom_score
        elif score_field == "median":
            data[r.residue_seq_id] = r.median_atom_score
        elif score_field == "max":
            data[r.residue_seq_id] = r.max_atom_score
        else:
            raise ValueError(
                f"Unknown score_field '{score_field}'. "
                "Choose from: musem, min, median, max."
            )
    return data


def extract_bfactors(
    structure_path: str,
    chain_id: Optional[str] = None,
    per_atom: bool = False,
    atom_name: str = "CA",
) -> ResidueDict:
    """
    Extract per-residue B-factors from a PDB / mmCIF file

    Args:
        structure_path: Path to PDB/mmCIF file
        chain_id: If given restrict to this chain
        per_atom: If True return the mean B-factor over all heavy atoms
        atom_name: Representative atom to use when ``per_atom=False`` Default ``'CA'``
    Returns:
        Dict mapping residue sequence number to B-factor
    """
    structure = gemmi.read_structure(str(structure_path))
    data: ResidueDict = {}
    model = structure[0]
    for chain in model:
        if chain_id is not None and chain.name != chain_id:
            continue
        for residue in chain:
            seq_id = residue.seqid.num
            if per_atom:
                bvals = [
                    atom.b_iso
                    for atom in residue
                    if atom.element not in (gemmi.Element("H"), gemmi.Element("D"))
                ]
                if bvals:
                    data[seq_id] = float(np.mean(bvals))
            else:
                chosen = None
                for atom in residue:
                    if atom.name == atom_name:
                        chosen = atom
                        break
                if chosen is None:
                    for atom in residue:
                        if atom.element not in (
                            gemmi.Element("H"), gemmi.Element("D")
                        ):
                            chosen = atom
                            break
                if chosen is not None:
                    data[seq_id] = float(chosen.b_iso)
    return data

def write_scored_pdb(
    result: MUSEResult,
    structure_path: str,
    output_path: str,
    score_level: str = "residue",
    score_field: str = "musem",
    score_scale: float = 100.0,
    missing_value: float = 0.0,
) -> None:
    """
    Write a PDB file with MUSE scores substituted into the B-factor column

    Args:
        result: MUSEResult
        structure_path: Path to the original PDB/mmCIF  file
        output_path: Destination path
        score_level: Granularity of scores
            * ``'residue'`` (default) — all atoms of a residue receive the
              residue's MUSEm aggregate score
            * ``'atom'`` — each atom receives its own per-atom MUSE score

        score_field: Which score to embed.
            For ``score_level='residue'``: ``'musem'``, ``'min'``, ``'median'``, or ``'max'``
            For ``score_level='atom'``: ``'score'``, ``'score_positive'``, or``'score_negative'``

        score_scale: Multiply each score by this factor
        missing_value: B-factor value assigned to atoms not present in the MUSE results

    Raises:
        ValueError: If *score_level* or *score_field* is unrecognised
    """
    if score_level not in ("residue", "atom"):
        raise ValueError("score_level must be 'residue' or 'atom'.")

    structure = gemmi.read_structure(str(structure_path))

    if score_level == "residue":
        _valid = {"musem", "min", "median", "max"}
        if score_field not in _valid:
            raise ValueError(
                f"For residue-level scoring, score_field must be one of {_valid}."
            )
        score_map: Dict[Tuple[str, int, str], float] = {}
        for r in result.residue_scores:
            key = (r.chain_id, r.residue_seq_id, r.insertion_code)
            if score_field == "musem":
                score_map[key] = r.musem_score
            elif score_field == "min":
                score_map[key] = r.min_atom_score
            elif score_field == "median":
                score_map[key] = r.median_atom_score
            else:
                score_map[key] = r.max_atom_score
    else:
        _valid_atom = {"score", "score_positive", "score_negative"}
        if score_field not in _valid_atom:
            raise ValueError(
                f"For atom-level scoring, score_field must be one of {_valid_atom}."
            )
        atom_score_map: Dict[Tuple[str, int, str, str], float] = {}
        for a in result.atom_scores:
            key = (a.chain_id, a.residue_seq_id, a.insertion_code, a.atom_name)
            if score_field == "score":
                atom_score_map[key] = a.score
            elif score_field == "score_positive":
                atom_score_map[key] = a.score_positive
            else:
                atom_score_map[key] = a.score_negative

    model = structure[0]
    for chain in model:
        for residue in chain:
            seq_id = residue.seqid.num
            ins_code = str(residue.seqid.icode).strip()
            for atom in residue:
                if score_level == "residue":
                    key = (chain.name, seq_id, ins_code)
                    val = score_map.get(key, missing_value)
                else:
                    key = (chain.name, seq_id, ins_code, atom.name)
                    val = atom_score_map.get(key, missing_value)
                atom.b_iso = float(val * score_scale)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(out))
    print(structure[0][0][0][0].__str__())



_LEFT_COLORS = [
    "#4878CF",   # steel blue
    "#6ACC65",   # green
    "#D65F5F",   # red
    "#B47CC7",   # purple
]
_RIGHT_COLORS = [
    "#EE854A",   # orange
    "#C44E52",   # crimson
    "#8172B3",   # muted purple
    "#CCB974",   # ochre
]


def _build_colormap_for_bins(
    colormap_name: str,
    vmin: float,
    vmax: float,
) -> Tuple[mcolors.Normalize, cm.ScalarMappable]:
    """Return a (Normalize, ScalarMappable) pair for bin colouring"""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(colormap_name)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return norm, sm


def plot_residue_profile(
    datasets: Dict[str, ResidueDict],
    left_series: List[str],
    right_series: Optional[List[str]] = None,
    bin_color_series: Optional[str] = None,
    transforms: Optional[Dict[str, Tuple[str, float]]] = None,
    seq_range: Optional[Union[Tuple[int, int], List[int]]] = None,
    left_ylabel: str = "Score",
    right_ylabel: str = "Score",
    title: str = "Residue Profile",
    figsize: Tuple[float, float] = (14, 5),
    left_fill: bool = True,
    left_fill_alpha: float = 0.35,
    line_alpha: float = 0.9,
    bin_alpha: float = 0.25,
    bin_colormap: str = "RdYlGn",
    bin_vmin: Optional[float] = None,
    bin_vmax: Optional[float] = None,
    show_colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    left_colors: Optional[List[str]] = None,
    right_colors: Optional[List[str]] = None,
    edia_threshold_lines: bool = False,
) -> Figure:
    """
    Create a dual-Y-axis residue profile plot
    Plots one or more per-residue score series against residue sequence number

    Args:
        datasets: Mapping of series label to ``{seq_id: value}`` dict
        left_series: List of keys from *datasets* to plot on the left Y-axis
        right_series: Optional list of keys from *datasets* to plot on the right Y-axis
        bin_color_series: Key from *datasets* whose values drive the bin background colours
        transforms: Optional per-series transform specification.  Dict mapping
            series label to ``(method, scale)`` tuple.  See
            :func:`apply_transform` for available methods.  Example::

                transforms={
                    "snr": ("log1p", 1.0),
                    "pvalue": ("none", 1.0),
                }

        seq_range: Residue range filter.  Either:

            * ``(first, last)`` tuple — keeps residues with seq_id in
              ``[first, last]`` inclusive, e.g. ``seq_range=(50, 150)``.
            * ``[id1, id2, ...]`` list — keeps exactly those seq_ids,
              useful for non-contiguous selections like loop regions.
        left_ylabel: Y-axis label for the left axis
        right_ylabel: Y-axis label for the right axis
        title: Figure title
        figsize: Figure size as ``(width, height)`` in inches
        left_fill: If True draw a filled area under left-axis curves
        left_fill_alpha: Alpha for the filled area
        line_alpha: Alpha for line plots
        bin_alpha: Alpha for the background bin rectangles
        bin_colormap: Matplotlib colormap name for bin colouring. Default ``'RdYlGn'`` (red = low, green = high).
        bin_vmin: Lower bound for bin colour normalisation
        bin_vmax: Upper bound for bin colour normalisation
        show_colorbar: If True and *bin_color_series* is set, draw a colorbar for the bin colours
        colorbar_label: Label for the colorbar
        left_colors: Override the default colour cycle for left-axis series
        right_colors: Override the default colour cycle for right-axis series
        edia_threshold_lines: If True, draw horizontal dashed lines at the EDIA
            quality thresholds 0.4 and 0.8 on the left axis.  Useful when the
            left axis shows EDIA / MUSEm electron-density scores.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.  Call
        ``fig.savefig(...)`` or ``plt.show()`` to display / save

    Example::

        from muse.visualization import (
            plot_residue_profile, extract_residue_scores, extract_bfactors,
        )

        fig = plot_residue_profile(
            datasets={
                "EDIA":    extract_residue_scores(edia_result),
                "SNR":     extract_residue_scores(snr_result,  transforms={"SNR": ("log1p", 1.0)}),
                "B-factor": extract_bfactors("model.pdb"),
                "pvalue":  extract_residue_scores(pval_result),
            },
            left_series=["EDIA", "B-factor"],
            right_series=["SNR"],
            bin_color_series="pvalue",
            left_ylabel="MUSEm / B (Å²)",
            right_ylabel="SNR log(1+x)",
            edia_threshold_lines=True,
        )
        fig.savefig("profile.png", dpi=150, bbox_inches="tight")
    """
    if not left_series:
        raise ValueError("left_series must contain at least one dataset label.")
    for label in left_series + (right_series or []):
        if label not in datasets:
            raise ValueError(
                f"Series '{label}' not found in datasets. "
                f"Available: {list(datasets.keys())}"
            )
    if bin_color_series is not None and bin_color_series not in datasets:
        raise ValueError(
            f"bin_color_series '{bin_color_series}' not found in datasets."
        )

    transforms = transforms or {}
    right_series = right_series or []
    lc = left_colors or _LEFT_COLORS
    rc = right_colors or _RIGHT_COLORS

    all_seqs: List[int] = sorted(
        {seq for series in datasets.values() for seq in series}
    )
    if seq_range is not None:
        if isinstance(seq_range, (list, set, frozenset)):
            _seq_set = set(seq_range)
            all_seqs = [s for s in all_seqs if s in _seq_set]
        else:
            all_seqs = [s for s in all_seqs if seq_range[0] <= s <= seq_range[1]]
    if not all_seqs:
        raise ValueError("No residue positions remain after applying seq_range.")

    def _get_values(label: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (seq_array, value_array) with NaN for missing positions"""
        d = datasets[label]
        seqs = np.array(all_seqs, dtype=float)
        vals = np.array([d.get(s, np.nan) for s in all_seqs], dtype=float)
        if label in transforms:
            method, scale = transforms[label]
            finite = np.isfinite(vals)
            vals[finite] = apply_transform(vals[finite], method, scale)
        return seqs, vals

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx() if right_series else None

    x_arr = np.array(all_seqs, dtype=float)
    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    bin_width = 1.0  # one residue per bin

    if bin_color_series is not None:
        bin_data = datasets[bin_color_series]
        raw_bin_vals = np.array([bin_data.get(s, np.nan) for s in all_seqs])
        if bin_color_series in transforms:
            method, scale = transforms[bin_color_series]
            finite = np.isfinite(raw_bin_vals)
            raw_bin_vals[finite] = apply_transform(
                raw_bin_vals[finite], method, scale
            )

        finite_bin = raw_bin_vals[np.isfinite(raw_bin_vals)]
        _vmin = bin_vmin if bin_vmin is not None else (
            float(finite_bin.min()) if len(finite_bin) else 0.0
        )
        _vmax = bin_vmax if bin_vmax is not None else (
            float(finite_bin.max()) if len(finite_bin) else 1.0
        )
        norm, sm = _build_colormap_for_bins(bin_colormap, _vmin, _vmax)

        for seq, bval in zip(all_seqs, raw_bin_vals):
            if not np.isfinite(bval):
                continue
            colour = sm.cmap(norm(bval))
            rect = mpatches.Rectangle(
                (seq - 0.5, 0), bin_width, 1,
                transform=ax_left.get_xaxis_transform(),
                color=colour, alpha=bin_alpha, linewidth=0, zorder=0,
            )
            ax_left.add_patch(rect)

        if show_colorbar:
            cb = fig.colorbar(sm, ax=ax_left, pad=0.01 if ax_right is None else 0.12,
                               shrink=0.9, aspect=30)
            cb.set_label(
                colorbar_label if colorbar_label is not None else bin_color_series,
                fontsize=9,
            )

    if edia_threshold_lines:
        for thresh, label_text in [(0.4, "0.4"), (0.8, "0.8")]:
            ax_left.axhline(thresh, color="grey", linewidth=0.8,
                            linestyle="--", alpha=0.6, zorder=1)
            ax_left.text(x_min, thresh, label_text, fontsize=7,
                         color="grey", va="bottom", ha="left")

    for idx, label in enumerate(left_series):
        colour = lc[idx % len(lc)]
        seqs, vals = _get_values(label)
        valid = np.isfinite(vals)
        if left_fill and idx == 0:
            ax_left.fill_between(
                seqs[valid], vals[valid], alpha=left_fill_alpha, color=colour,
                zorder=2,
            )
        ax_left.plot(
            seqs[valid], vals[valid],
            color=colour, linewidth=1.5, alpha=line_alpha,
            label=label, zorder=3,
        )

    ax_left.set_xlabel("Residue sequence", fontsize=11)
    ax_left.set_ylabel(left_ylabel, fontsize=11)
    ax_left.set_xlim(x_min - 1, x_max + 1)
    ax_left.tick_params(axis="both", labelsize=9)

    if ax_right is not None:
        for idx, label in enumerate(right_series):
            colour = rc[idx % len(rc)]
            seqs, vals = _get_values(label)
            valid = np.isfinite(vals)
            ax_right.plot(
                seqs[valid], vals[valid],
                color=colour, linewidth=1.5, linestyle="--", alpha=line_alpha,
                label=label, zorder=3,
            )
        ax_right.set_ylabel(right_ylabel, fontsize=11)
        ax_right.tick_params(axis="y", labelsize=9)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    if ax_right is not None:
        lines_r, labels_r = ax_right.get_legend_handles_labels()
        ax_left.legend(
            lines_l + lines_r, labels_l + labels_r,
            fontsize=9, loc="upper right", framealpha=0.85,
        )
    elif lines_l:
        ax_left.legend(lines_l, labels_l, fontsize=9, loc="upper right",
                        framealpha=0.85)

    ax_left.set_title(title, fontsize=12, pad=8)
    fig.tight_layout()
    return fig


def edia_colormap() -> mcolors.LinearSegmentedColormap:
    """
    Return a matplotlib colormap matching the EDIA quality colour scale

    * 0.0 – 0.4  → red   (substantial inconsistencies)
    * 0.4 – 0.8  → pink  (minor inconsistencies)
    * 0.8 – 1.2  → blue  (well covered)

    Returns:
        A :class:`~matplotlib.colors.LinearSegmentedColormap` named
        ``'edia'``.
    """
    colours = [
        (0.0,  "#D62728"),   # red
        (0.4 / 1.2, "#FF9896"),  # pink
        (0.8 / 1.2, "#AEC7E8"),  # light blue
        (1.0,  "#1F77B4"),   # blue
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "edia", [(v, c) for v, c in colours]
    )
    return cmap


def pvalue_colormap() -> mcolors.LinearSegmentedColormap:
    """Return a colormap suited to p-value / significance maps [0, 1]

    * 0.0 → red   (low confidence / noise)
    * 0.5 → white / yellow
    * 1.0 → green (high confidence / significant density)

    Returns:
        A :class:`~matplotlib.colors.LinearSegmentedColormap` named
        ``'pvalue'``.
    """
    colours = [
        (0.0,  "#D62728"),   # red
        (0.25, "#FF7F0E"),   # orange
        (0.5,  "#FFDD71"),   # yellow
        (0.75, "#98DF8A"),   # light green
        (1.0,  "#2CA02C"),   # green
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "pvalue", [(v, c) for v, c in colours]
    )


_CPK_COLORS: Dict[str, str] = {
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "S": "#FFFF30",
    "P": "#FF8000",
    "F": "#90E050",
    "CL": "#1FF01F",
    "BR": "#A62929",
    "I":  "#940094",
    "MG": "#8AFF00",
    "CA": "#3DFF00",
    "ZN": "#7D80B0",
    "FE": "#E06633",
    "CU": "#FF8040",
    "MN": "#9C7AC7",
    "CO": "#F090A0",
    "H":  "#FFFFFF",
}

_WATER_NAMES: frozenset = frozenset({"HOH", "WAT", "DOD", "H2O", "SOL"})


def _cpk(element: str) -> str:
    return _CPK_COLORS.get(element.upper(), "#888888")


def _draw_bond_typed(
    ax,
    p1: np.ndarray,
    p2: np.ndarray,
    bond_type: str,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    """Draw one bond between 2-D points p1 and p2 with correct bond-order styling.

    bond_type must be one of ``'single'``, ``'double'``, ``'triple'``,
    ``'aromatic'``.  Double bonds are drawn as two parallel offset lines;
    aromatic bonds as a solid line plus a shorter dashed inner line; triple
    bonds as three parallel lines.
    """
    dx, dy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return

    # Perpendicular unit vector scaled to a small offset
    offset = min(0.12, length * 0.12)
    px, py = -dy / length * offset, dx / length * offset

    def _line(shift_x, shift_y, ls="-", lw_scale=1.0, alpha=1.0):
        ax.plot(
            [p1[0] + shift_x, p2[0] + shift_x],
            [p1[1] + shift_y, p2[1] + shift_y],
            color=color, linewidth=linewidth * lw_scale,
            linestyle=ls, solid_capstyle="round",
            zorder=zorder, alpha=alpha,
        )

    if bond_type == "double":
        _line(+px, +py, lw_scale=0.75)
        _line(-px, -py, lw_scale=0.75)
    elif bond_type == "triple":
        _line(0.0, 0.0, lw_scale=0.6)
        _line(+px * 1.6, +py * 1.6, lw_scale=0.5)
        _line(-px * 1.6, -py * 1.6, lw_scale=0.5)
    elif bond_type == "aromatic":
        _line(0.0, 0.0)                              # solid outer
        _line(+px, +py, ls="--", lw_scale=0.55, alpha=0.7)  # dashed inner
    else:  # single (or unknown)
        _line(0.0, 0.0)


def plot_ligand_density_support(
    atom_scores: Union[List[AtomScore], Dict[str, float]],
    structure_path: str,
    chain_id: str,
    residue_seq_id: int,
    residue_name: Optional[str] = None,
    insertion_code: str = "",
    score_key: str = "score_positive",
    projection: str = "pca",
    grid_resolution: float = 0.08,
    atom_sigma: float = 0.7,
    n_contour_levels: int = 6,
    good_threshold: float = 0.8,
    poor_threshold: float = 0.4,
    padding: float = 2.0,
    figsize: Tuple[float, float] = (7, 7),
    title: Optional[str] = None,
    bg_color: str = "#ebebeb",
    good_color: str = "#1a6b1a",
    poor_color: str = "#e05080",
    bond_color: str = "#333333",
    bond_linewidth: float = 2.0,
    show_atom_labels: bool = True,
    label_carbon: bool = False,
) -> Figure:
    """
    Overlay per-atom density support on a 2-D projected small-molecule structure

    Args:
        atom_scores: Per-atom scores for the target residue.  Either:

            * A ``List[AtomScore]`` from :func:`~muse.pipeline.run_muse` —
              the field selected by *score_key* is used as the score value.
            * A ``Dict[str, float]`` mapping atom name → custom score value
              (e.g. from a separate analysis).

        structure_path: Path to the PDB / mmCIF coordinate file.  Atom 3-D
            positions and bond connectivity are read from here.
        chain_id: Chain identifier of the target residue.
        residue_seq_id: Residue sequence number.
        residue_name: Optional residue name for disambiguation.  If ``None``
            the first residue matching *chain_id* and *residue_seq_id* is used.
        insertion_code: Insertion code.  Default empty string.
        score_key: Which field to extract from ``AtomScore`` objects when
            *atom_scores* is a list.  One of ``'score'``, ``'score_positive'``
            (default), or ``'score_negative'``.
        projection: Fallback 3-D → 2-D projection used only when RDKit is not
            installed.  When RDKit **is** available the 2-D layout is computed
            automatically from the molecule topology via
            ``AllChem.Compute2DCoords()`` (proper chemical diagram, no
            external SMILES required).  Options:

            * ``'pca'`` (default) — principal-component projection onto the
              plane of maximum variance.
            * ``'xy'``, ``'xz'``, ``'yz'`` — drop the Z, Y, or X axis
              respectively.

        grid_resolution: Grid spacing in Å for the Gaussian density field.
            Smaller values give smoother contours but take longer.  Default 0.08.
        atom_sigma: Width (σ) of the 2-D Gaussian kernel placed at each atom,
            in Å.  Default 0.7.
        n_contour_levels: Number of contour levels for the green blobs.
            Default 6.
        good_threshold: Atoms with score ≥ this value contribute to the green
            contours.  Default 0.8.
        poor_threshold: Atoms with score < this value receive a dashed pink
            circle indicating missing density.  Default 0.4.
        padding: Extra space (Å) added around the molecule extent on all sides.
            Default 2.0.
        figsize: Figure size in inches.  Default ``(7, 7)``.
        title: Figure title.  Defaults to ``"<residue_name> <seq_id> — density
            support"``.
        bg_color: Background colour of the axes.  Default light grey.
        good_color: Colour for the positive-density contours.  Default dark
            green ``'#1a6b1a'``.
        poor_color: Colour for the missing-density dashed circles.  Default
            pink ``'#e05080'``.
        bond_color: Colour for bond lines.  Default near-black.
        bond_linewidth: Line width for bonds in points.  Default 2.0.
        show_atom_labels: If True (default), draw element symbols for all
            non-carbon heteroatoms.  Carbon atoms are labelled only when
            *label_carbon* is also True.
        label_carbon: If True, draw 'C' labels on carbon atoms too.
            Default False.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.

    Example::

        from muse import run_muse
        from muse.visualization import plot_ligand_density_support

        result = run_muse("2fofc.ccp4", "model.pdb", resolution=1.8)

        # Extract atom scores for ligand LIG chain A residue 401
        lig_scores = [
            a for a in result.atom_scores
            if a.chain_id == "A" and a.residue_seq_id == 401
        ]
        fig = plot_ligand_density_support(
            lig_scores, "model.pdb", chain_id="A", residue_seq_id=401,
        )
        fig.savefig("lig401_density.png", dpi=150, bbox_inches="tight")
    """
    # ── Build score lookup {atom_name: value} ────────────────────────────────
    if isinstance(atom_scores, dict):
        score_lookup: Dict[str, float] = {k: float(v) for k, v in atom_scores.items()}
    else:
        _valid_keys = {"score", "score_positive", "score_negative"}
        if score_key not in _valid_keys:
            raise ValueError(
                f"score_key must be one of {_valid_keys}, got '{score_key}'."
            )
        score_lookup = {}
        for a in atom_scores:
            val = getattr(a, score_key)
            score_lookup[a.atom_name] = float(val)

    # ── Load atom positions for the target residue ───────────────────────────
    structure = gemmi.read_structure(str(structure_path))
    model = structure[0]

    target_residue: Optional[gemmi.Residue] = None
    for chain in model:
        if chain.name != chain_id:
            continue
        for residue in chain:
            if residue.seqid.num != residue_seq_id:
                continue
            if residue_name and residue.name != residue_name:
                continue
            ins = str(residue.seqid.icode).strip()
            if ins != insertion_code.strip():
                continue
            target_residue = residue
            break
        if target_residue is not None:
            break

    if target_residue is None:
        raise ValueError(
            f"Residue {chain_id}/{residue_seq_id}{insertion_code} not found "
            f"in {structure_path}."
        )

    # Collect atom data (skip H/D)
    atom_names: List[str] = []
    elements: List[str] = []
    positions_3d: List[np.ndarray] = []

    for atom in target_residue:
        if atom.element in (gemmi.Element("H"), gemmi.Element("D")):
            continue
        atom_names.append(atom.name)
        elements.append(atom.element.name)
        positions_3d.append(np.array([atom.pos.x, atom.pos.y, atom.pos.z]))

    if not positions_3d:
        raise ValueError("No heavy atoms found in target residue.")

    coords3 = np.array(positions_3d)   # (N, 3)

    # ── Detect covalent bonds from 3-D structure (distance criterion) ─────────
    bonds: List[Tuple[int, int]] = []
    for i, (pos_i, elem_i) in enumerate(zip(positions_3d, elements)):
        ri = float(gemmi.Element(elem_i).covalent_r)
        for j in range(i + 1, len(positions_3d)):
            rj = float(gemmi.Element(elements[j]).covalent_r)
            if float(np.linalg.norm(pos_i - positions_3d[j])) <= ri + rj + 0.4:
                bonds.append((i, j))

    # ── 2-D coordinates: build RDKit mol from PDB topology → Compute2DCoords ──
    coords2: np.ndarray
    bond_types: List[str] = ["single"] * len(bonds)
    _layout_ok = False

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Build a writable RDKit mol from the PDB heavy atoms.
        # Atom i in the RDKit mol corresponds to atom_names[i] / elements[i],
        # so no index mapping is needed after coordinate generation.
        rwmol = Chem.RWMol()
        for elem in elements:
            rwmol.AddAtom(Chem.Atom(elem))

        # Add all detected bonds as SINGLE first; sanitization below will
        # promote them to AROMATIC / DOUBLE where topology demands it.
        for bi, bj in bonds:
            rwmol.AddBond(bi, bj, Chem.rdchem.BondType.SINGLE)

        # Sanitize to perceive aromaticity, hybridization, conjugation.
        # Ligand valences sometimes violate RDKit defaults, so we catch any
        # failure and continue with the partial sanitization result.
        try:
            Chem.SanitizeMol(rwmol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    rwmol,
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                )
            except Exception:
                pass  # proceed with single bonds only

        mol = rwmol.GetMol()

        # Compute proper chemical 2-D coordinates (ring systems, bond angles).
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()

        # Direct coordinate extraction — atom ordering is preserved.
        coords2 = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                for i in range(mol.GetNumAtoms())
            ]
        )

        # Pull bond types from the sanitized mol for correct visual rendering.
        _BT = Chem.rdchem.BondType
        _type_map = {
            _BT.DOUBLE:   "double",
            _BT.TRIPLE:   "triple",
            _BT.AROMATIC: "aromatic",
        }
        bond_types = []
        for bi, bj in bonds:
            rdkit_bond = mol.GetBondBetweenAtoms(bi, bj)
            if rdkit_bond is not None:
                bond_types.append(_type_map.get(rdkit_bond.GetBondType(), "single"))
            else:
                bond_types.append("single")

        _layout_ok = True

    except ImportError:
        import warnings
        warnings.warn(
            "RDKit is not installed; falling back to 3-D projection. "
            "Install with: pip install rdkit",
            stacklevel=2,
        )

    if not _layout_ok:
        # ── 3-D projection fallback (no RDKit) ───────────────────────────────
        centroid = coords3.mean(axis=0)
        centered = coords3 - centroid

        if projection == "pca":
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            coords2 = centered @ Vt[:2].T
        elif projection == "xy":
            coords2 = centered[:, :2]
        elif projection == "xz":
            coords2 = centered[:, [0, 2]]
        elif projection == "yz":
            coords2 = centered[:, [1, 2]]
        else:
            raise ValueError(
                f"projection must be 'pca', 'xy', 'xz', or 'yz', "
                f"got '{projection}'."
            )

    # ── Build Gaussian field ─────────────────────────────────────────────────
    x0 = coords2[:, 0].min() - padding
    x1 = coords2[:, 0].max() + padding
    y0 = coords2[:, 1].min() - padding
    y1 = coords2[:, 1].max() + padding

    nx = max(int((x1 - x0) / grid_resolution) + 1, 2)
    ny = max(int((y1 - y0) / grid_resolution) + 1, 2)
    xi = np.linspace(x0, x1, nx)
    yi = np.linspace(y0, y1, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    field = np.zeros_like(Xi)
    two_sig2 = 2.0 * atom_sigma ** 2

    for idx, aname in enumerate(atom_names):
        s = score_lookup.get(aname, 0.0)
        if s <= 0.0:
            continue
        ax_, ay_ = coords2[idx, 0], coords2[idx, 1]
        dist_sq = (Xi - ax_) ** 2 + (Yi - ay_) ** 2
        field += s * np.exp(-dist_sq / two_sig2)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")

    # Contour fill + lines for positive-density atoms
    field_max = field.max()
    if field_max > 1e-6 and n_contour_levels > 0:
        # Only draw contours above a small fraction of max (avoids flat noise)
        level_min = field_max * 0.05
        levels = np.linspace(level_min, field_max * 0.98, n_contour_levels + 1)
        # Filled contours (progressively darker green)
        cmap_contour = mcolors.LinearSegmentedColormap.from_list(
            "_green_density",
            [(0.0, "#d4f0d4"), (0.5, "#5ab55a"), (1.0, good_color)],
        )
        ax.contourf(Xi, Yi, field, levels=levels, cmap=cmap_contour,
                    alpha=0.85, zorder=1)
        ax.contour(Xi, Yi, field, levels=levels[1:],
                   colors=good_color, linewidths=0.6, alpha=0.7, zorder=2)

    # ── Bond lines (single / double / aromatic / triple) ─────────────────────
    for (i, j), btype in zip(bonds, bond_types):
        _draw_bond_typed(ax, coords2[i], coords2[j], btype,
                         bond_color, bond_linewidth, zorder=3)

    # ── Atom markers and labels ───────────────────────────────────────────────
    for idx, (aname, elem) in enumerate(zip(atom_names, elements)):
        ax_, ay_ = coords2[idx, 0], coords2[idx, 1]
        score_val = score_lookup.get(aname, np.nan)
        is_good = np.isfinite(score_val) and score_val >= good_threshold
        is_poor = np.isfinite(score_val) and score_val < poor_threshold

        # Dashed circle for missing-density atoms
        if is_poor:
            circle = plt.Circle(
                (ax_, ay_), radius=atom_sigma * 1.4,
                fill=False, linestyle="--", linewidth=1.6,
                edgecolor=poor_color, alpha=0.9, zorder=4,
            )
            ax.add_patch(circle)

        # Atom dot (colored by element for heteroatoms, semi-transparent for C)
        dot_color = _cpk(elem)
        dot_size = 80 if elem.upper() != "C" else 40
        dot_alpha = 1.0 if elem.upper() != "C" else 0.4
        ax.scatter([ax_], [ay_], s=dot_size, c=dot_color, zorder=5,
                   edgecolors="white", linewidths=0.5, alpha=dot_alpha)

        # Atom label
        if show_atom_labels:
            is_carbon = elem.upper() == "C"
            if not is_carbon or label_carbon:
                ax.text(
                    ax_ + atom_sigma * 0.6, ay_ + atom_sigma * 0.6,
                    elem.upper(), fontsize=8, fontweight="bold",
                    color=_cpk(elem), zorder=6,
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.6),
                )

    # ── Legend patches ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=good_color, alpha=0.8,
                       label=f"MUSE ≥ {good_threshold} (well-supported)"),
        mpatches.Patch(facecolor="none", edgecolor=poor_color, linestyle="--",
                       label=f"MUSE < {poor_threshold} (missing density)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
              framealpha=0.85)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title or f"{target_residue.name} {residue_seq_id} — density support",
        fontsize=12, pad=8,
    )
    fig.tight_layout()
    return fig

def plot_water_support(
    result: MUSEResult,
    threshold: float = 0.5,
    chain_id: Optional[str] = None,
    score_field: str = "musem",
    colormap: str = "RdYlGn",
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Water Molecule Density Support",
    label_near_threshold: float = 0.1,
    max_label: int = 30,
) -> Figure:
    """Visualise per-water density support scores for a complete structure.

    Produces a two-panel figure:

    * **Top panel** — sorted scatter / rank plot showing all water scores in
      descending order, colour-coded by score value, with a horizontal
      threshold line.  Waters below the threshold are marked distinctly.
    * **Bottom panel** — histogram of the score distribution with the
      threshold indicated by a vertical line and keep / remove counts
      annotated.

    Designed to handle structures with 400+ water molecules without visual
    clutter.

    Args:
        result: Completed :class:`~muse.scoring.MUSEResult`.
        threshold: Score threshold below which waters are considered
            unresolved (and would be removed by
            :func:`filter_waters_by_score`).  Default 0.5.
        chain_id: If given, restrict to waters on this chain.
        score_field: Which residue-level score to use.  One of ``'musem'``
            (default), ``'min'``, ``'median'``, ``'max'``.
        colormap: Matplotlib colormap name for score colouring.
            Default ``'RdYlGn'``.
        figsize: Figure size.  Auto-scaled from water count if ``None``.
        title: Figure title.
        label_near_threshold: Waters whose score falls within this distance of
            *threshold* are labelled with their sequence ID in the rank plot.
            Set to 0 to disable auto-labelling.
        max_label: Maximum number of labels to draw in the rank plot to avoid
            overcrowding.  Default 30.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    # ── Collect water residue scores ─────────────────────────────────────────
    water_data: List[Tuple[int, str, float]] = []   # (seq_id, chain, score)
    for r in result.residue_scores:
        if r.residue_name.upper() not in _WATER_NAMES:
            continue
        if chain_id is not None and r.chain_id != chain_id:
            continue
        if score_field == "musem":
            val = r.musem_score
        elif score_field == "min":
            val = r.min_atom_score
        elif score_field == "median":
            val = r.median_atom_score
        elif score_field == "max":
            val = r.max_atom_score
        else:
            raise ValueError(
                f"score_field must be one of musem/min/median/max, got '{score_field}'."
            )
        water_data.append((r.residue_seq_id, r.chain_id, val))

    if not water_data:
        raise ValueError("No water residues found in the result.")

    # Sort by score descending
    water_data.sort(key=lambda x: x[2], reverse=True)
    seq_ids = np.array([w[0] for w in water_data])
    scores = np.array([w[2] for w in water_data])
    n_waters = len(scores)
    n_keep = int((scores >= threshold).sum())
    n_remove = n_waters - n_keep
    ranks = np.arange(1, n_waters + 1)

    # Auto figsize
    if figsize is None:
        width = max(10, min(20, n_waters * 0.04 + 8))
        figsize = (width, 7)

    # ── Colormap ─────────────────────────────────────────────────────────────
    vmin, vmax = float(scores.min()), float(scores.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(colormap)
    colors_arr = cmap(norm(scores))

    fig, (ax_rank, ax_hist) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    # ── Rank plot (top panel) ─────────────────────────────────────────────────
    ax_rank.axhline(threshold, color="#555555", linewidth=1.5, linestyle="--",
                    alpha=0.9, zorder=1, label=f"Threshold = {threshold:.2f}")

    # Shade removal region
    ax_rank.axhspan(vmin - 0.05, threshold, color="#FFDDDD", alpha=0.35, zorder=0)

    # Scatter all waters
    ax_rank.scatter(
        ranks, scores, c=colors_arr, s=20, zorder=3,
        linewidths=0.0, alpha=0.85,
    )

    # Label waters near the threshold
    if label_near_threshold > 0:
        near_mask = np.abs(scores - threshold) <= label_near_threshold
        near_idx = np.where(near_mask)[0]
        if len(near_idx) > max_label:
            # Keep the closest ones
            near_idx = near_idx[
                np.argsort(np.abs(scores[near_idx] - threshold))[:max_label]
            ]
        for idx in near_idx:
            ax_rank.annotate(
                str(seq_ids[idx]),
                xy=(ranks[idx], scores[idx]),
                xytext=(3, 4), textcoords="offset points",
                fontsize=6, color="#333333", alpha=0.8,
            )

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_rank, pad=0.01, shrink=0.9, aspect=25)
    cb.set_label(f"MUSEm score ({score_field})", fontsize=9)

    ax_rank.set_xlim(0, n_waters + 1)
    ax_rank.set_xlabel("Rank (highest score first)", fontsize=10)
    ax_rank.set_ylabel(f"Score ({score_field})", fontsize=10)
    ax_rank.set_title(title, fontsize=12, pad=8)
    ax_rank.legend(fontsize=9, loc="upper right")
    ax_rank.annotate(
        f"Keep: {n_keep}  |  Remove: {n_remove}  |  Total: {n_waters}",
        xy=(0.02, 0.04), xycoords="axes fraction",
        fontsize=9, color="#333333",
    )

    # ── Histogram (bottom panel) ──────────────────────────────────────────────
    n_bins = max(20, min(60, n_waters // 5))
    ax_hist.hist(
        scores[scores >= threshold], bins=n_bins,
        color=cmap(0.85), alpha=0.8, label="Keep",
        range=(vmin, vmax),
    )
    ax_hist.hist(
        scores[scores < threshold], bins=n_bins,
        color=cmap(0.15), alpha=0.8, label="Remove",
        range=(vmin, vmax),
    )
    ax_hist.axvline(threshold, color="#555555", linewidth=1.5, linestyle="--")
    ax_hist.set_xlabel(f"Score ({score_field})", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Water filtering: remove low-score waters and write new PDB
# ---------------------------------------------------------------------------

def filter_waters_by_score(
    result: MUSEResult,
    structure_path: str,
    output_path: str,
    threshold: float = 0.5,
    score_field: str = "musem",
    chain_id: Optional[str] = None,
) -> Tuple[int, int]:
    """Remove low-confidence water molecules and write a filtered PDB / mmCIF.

    Waters whose per-residue aggregated score (from a p-value or
    electron-density map run) falls below *threshold* are deleted from the
    model.  The filtered structure is written to *output_path* in PDB format.

    Args:
        result: Completed :class:`~muse.scoring.MUSEResult` computed from a
            p-value (or other quality) map applied to the structure that
            contains the water molecules.
        structure_path: Path to the original PDB / mmCIF coordinate file.
        output_path: Destination path for the filtered structure.  Must end in
            ``.pdb`` or ``.cif``; the format is chosen automatically.
        threshold: Waters with score *strictly below* this value are removed.
            Default 0.5.  Tune based on your p-value / MUSE score scale:

            * For p-value MUSEm: 0.3 – 0.6 is a typical range.
            * For electron-density MUSEm: use the EDIA threshold of 0.4–0.8.

        score_field: Which residue-level score to compare against *threshold*.
            One of ``'musem'`` (default), ``'min'``, ``'median'``, ``'max'``.
        chain_id: If given, only waters on this chain are considered for
            removal.  Waters on other chains are always kept.

    Returns:
        ``(n_kept, n_removed)`` — counts of water residues retained and
        deleted.

    Raises:
        ValueError: If *score_field* is unrecognised.

    Example::

        from muse import run_muse
        from muse.visualization import filter_waters_by_score

        pval_result = run_muse(
            "pvalue_map.ccp4", "model.pdb", resolution=2.0,
            config=pvalue_map_config(),
        )
        n_kept, n_removed = filter_waters_by_score(
            pval_result, "model.pdb", "model_filtered.pdb", threshold=0.4,
        )
        print(f"Removed {n_removed} water molecules, kept {n_kept}.")
    """
    _valid = {"musem", "min", "median", "max"}
    if score_field not in _valid:
        raise ValueError(
            f"score_field must be one of {_valid}, got '{score_field}'."
        )

    # ── Build set of (chain, seq_id, ins_code) to REMOVE ─────────────────────
    remove_keys: set = set()
    for r in result.residue_scores:
        if r.residue_name.upper() not in _WATER_NAMES:
            continue
        if chain_id is not None and r.chain_id != chain_id:
            continue
        if score_field == "musem":
            val = r.musem_score
        elif score_field == "min":
            val = r.min_atom_score
        elif score_field == "median":
            val = r.median_atom_score
        else:
            val = r.max_atom_score

        if val < threshold:
            remove_keys.add((r.chain_id, r.residue_seq_id, r.insertion_code))

    # ── Load structure and delete flagged water residues ──────────────────────
    structure = gemmi.read_structure(str(structure_path))
    model = structure[0]

    n_removed = 0
    for chain in model:
        residues_to_keep: List[gemmi.Residue] = []
        for residue in chain:
            if residue.name.upper() not in _WATER_NAMES:
                residues_to_keep.append(residue)
                continue
            seq_id = residue.seqid.num
            ins = str(residue.seqid.icode).strip()
            key = (chain.name, seq_id, ins)
            if key in remove_keys:
                n_removed += 1
            else:
                residues_to_keep.append(residue)

        # Replace chain contents
        while len(chain) > 0:
            chain.__delitem__(0)
        for res in residues_to_keep:
            chain.add_residue(res)

    # Count waters remaining in the kept set
    n_kept = sum(
        1
        for chain in model
        for residue in chain
        if residue.name.upper() in _WATER_NAMES
        and (chain_id is None or chain.name == chain_id)
    )

    # ── Write output ──────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".cif", ".mmcif"):
        structure.make_mmcif_document().write_file(str(out))
    else:
        structure.write_pdb(str(out))

    return n_kept, n_removed
