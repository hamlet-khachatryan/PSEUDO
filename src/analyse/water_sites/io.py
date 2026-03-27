from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from analyse.water_sites.clustering import WaterSite
from analyse.water_sites.metrics import WaterSiteConsistency
from analyse.water_sites.occupancy import SiteOccupancy
from analyse.water_sites.site_snr import PerStructureSiteSNR, SNRStats


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _f(v: Optional[float], precision: int = 6) -> str:
    """Format a float; return empty string for None or NaN."""
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return f"{v:.{precision}f}"


def _stats_cols(stats: Optional[SNRStats], prefix: str) -> Dict[str, str]:
    """Expand SNRStats into a dict of prefixed column-name → value strings."""
    if stats is None:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_median": "",
            f"{prefix}_min": "",
            f"{prefix}_max": "",
            f"{prefix}_std": "",
            f"{prefix}_n_points": "",
        }
    return {
        f"{prefix}_mean": _f(stats.mean),
        f"{prefix}_median": _f(stats.median),
        f"{prefix}_min": _f(stats.min),
        f"{prefix}_max": _f(stats.max),
        f"{prefix}_std": _f(stats.std),
        f"{prefix}_n_points": str(stats.n_points),
    }


def _write_csv(
    output_path: Path,
    fieldnames: List[str],
    rows: List[Dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary CSV  (one row per water site)
# ---------------------------------------------------------------------------

_SUMMARY_FIELDS = [
    "site_id",
    "centroid_x", "centroid_y", "centroid_z",
    "site_radius",
    "n_structures_total", "n_structures_with_water", "water_frequency",
    # Across-screen aggregation of per-structure mean-SNR (site-radius sphere)
    "snr_site_r_mean", "snr_site_r_median",
    "snr_site_r_min", "snr_site_r_max", "snr_site_r_std",
    # Across-screen aggregation of per-structure mean-SNR (1.5 Å sphere)
    "snr_1_5_mean", "snr_1_5_median",
    "snr_1_5_min", "snr_1_5_max", "snr_1_5_std",
    # Across-screen aggregation of per-structure mean-SNR (2.0 Å sphere)
    "snr_2_0_mean", "snr_2_0_median",
    "snr_2_0_min", "snr_2_0_max", "snr_2_0_std",
    # Consistency metrics
    "snr_cv", "snr_above_1_fraction", "consistency_score",
    "snr_tstat", "snr_pvalue",
    "most_common_occupancy",
]


def _means_for(
    snrs: List[PerStructureSiteSNR], attr: str
) -> List[Optional[float]]:
    """Extract the within-sphere mean SNR for *attr* from each record."""
    return [
        getattr(s, attr).mean if getattr(s, attr) is not None else None
        for s in snrs
    ]


def _agg_across_screen(
    values: List[Optional[float]],
) -> Dict[str, str]:
    """Aggregate a list of per-structure scalar values across the screen."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return {"mean": "", "median": "", "min": "", "max": "", "std": ""}
    arr = np.array(valid, dtype=np.float64)
    return {
        "mean": _f(float(np.mean(arr))),
        "median": _f(float(np.median(arr))),
        "min": _f(float(np.min(arr))),
        "max": _f(float(np.max(arr))),
        "std": _f(float(np.std(arr))),
    }


def export_water_sites_summary(
    sites: List[WaterSite],
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
    consistencies: List[WaterSiteConsistency],
    n_total_structures: int,
    output_path: Path,
) -> None:
    """
    Write water_sites_summary.csv — one row per water site.

    For each radius the SNR columns contain the across-screen aggregation
    (mean/median/min/max/std) of the per-structure within-sphere mean SNR.
    """
    from collections import defaultdict

    snr_by_site: Dict[int, List[PerStructureSiteSNR]] = defaultdict(list)
    for s in per_structure_snr:
        snr_by_site[s.site_id].append(s)

    rows = []
    for site, cons in zip(sites, consistencies):
        site_snrs = snr_by_site[site.site_id]

        sr = _agg_across_screen(_means_for(site_snrs, "snr_site_radius"))
        r15 = _agg_across_screen(_means_for(site_snrs, "snr_1_5"))
        r20 = _agg_across_screen(_means_for(site_snrs, "snr_2_0"))

        rows.append({
            "site_id": site.site_id,
            "centroid_x": _f(site.centroid[0]),
            "centroid_y": _f(site.centroid[1]),
            "centroid_z": _f(site.centroid[2]),
            "site_radius": _f(site.radius),
            "n_structures_total": n_total_structures,
            "n_structures_with_water": site.n_waters,
            "water_frequency": _f(cons.water_frequency),
            "snr_site_r_mean": sr["mean"], "snr_site_r_median": sr["median"],
            "snr_site_r_min": sr["min"], "snr_site_r_max": sr["max"],
            "snr_site_r_std": sr["std"],
            "snr_1_5_mean": r15["mean"], "snr_1_5_median": r15["median"],
            "snr_1_5_min": r15["min"], "snr_1_5_max": r15["max"],
            "snr_1_5_std": r15["std"],
            "snr_2_0_mean": r20["mean"], "snr_2_0_median": r20["median"],
            "snr_2_0_min": r20["min"], "snr_2_0_max": r20["max"],
            "snr_2_0_std": r20["std"],
            "snr_cv": _f(cons.snr_cv),
            "snr_above_1_fraction": _f(cons.snr_above_1_fraction),
            "consistency_score": _f(cons.consistency_score),
            "snr_tstat": _f(cons.snr_tstat),
            "snr_pvalue": _f(cons.snr_pvalue),
            "most_common_occupancy": cons.most_common_occupancy,
        })

    _write_csv(output_path, _SUMMARY_FIELDS, rows)


# ---------------------------------------------------------------------------
# Per-structure CSV  (one row per (water site, structure) pair)
# ---------------------------------------------------------------------------

_PER_STRUCTURE_FIELDS = [
    "site_id", "stem", "has_water_modelled",
    # Within-sphere SNR stats for this structure (site-radius sphere)
    "snr_site_r_mean", "snr_site_r_median",
    "snr_site_r_min", "snr_site_r_max", "snr_site_r_std", "snr_site_r_n_points",
    # Within-sphere SNR stats (1.5 Å sphere)
    "snr_1_5_mean", "snr_1_5_median",
    "snr_1_5_min", "snr_1_5_max", "snr_1_5_std", "snr_1_5_n_points",
    # Within-sphere SNR stats (2.0 Å sphere)
    "snr_2_0_mean", "snr_2_0_median",
    "snr_2_0_min", "snr_2_0_max", "snr_2_0_std", "snr_2_0_n_points",
    # Occupancy
    "occupancy_type", "occupancy_residue_name",
    "occupancy_chain_id", "occupancy_seq_id", "occupancy_distance",
]


def export_water_sites_per_structure(
    per_structure_snr: List[PerStructureSiteSNR],
    per_structure_occ: List[SiteOccupancy],
    output_path: Path,
) -> None:
    """
    Write water_sites_per_structure.csv — one row per (water site, structure).

    SNR columns reflect the within-sphere aggregate statistics for that
    specific structure (mean/median/min/max/std/n_points of the raw SNR
    values sampled inside the sphere).
    """
    occ_index: Dict[tuple, SiteOccupancy] = {
        (o.site_id, o.stem): o for o in per_structure_occ
    }

    rows = []
    for snr in per_structure_snr:
        occ = occ_index.get((snr.site_id, snr.stem))
        row: Dict[str, str] = {
            "site_id": str(snr.site_id),
            "stem": snr.stem,
            "has_water_modelled": str(snr.has_water_modelled),
            **_stats_cols(snr.snr_site_radius, "snr_site_r"),
            **_stats_cols(snr.snr_1_5, "snr_1_5"),
            **_stats_cols(snr.snr_2_0, "snr_2_0"),
            "occupancy_type": occ.occupancy_type.value if occ else "",
            "occupancy_residue_name": occ.residue_name if occ else "",
            "occupancy_chain_id": occ.chain_id if occ else "",
            "occupancy_seq_id": str(occ.seq_id) if occ else "",
            "occupancy_distance": _f(occ.distance_to_centroid) if occ else "",
        }
        rows.append(row)

    _write_csv(output_path, _PER_STRUCTURE_FIELDS, rows)
