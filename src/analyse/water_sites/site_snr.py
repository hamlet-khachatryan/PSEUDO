from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np

from analyse.muse.grid_utils import enumerate_grid_points_in_sphere
from analyse.muse.io import load_map
from analyse.water_sites.alignment import SuperpositionResult, transform_to_structure
from analyse.water_sites.clustering import WaterSite
from analyse.water_sites.config import WaterSiteConfig


@dataclass
class SNRStats:
    """Aggregate statistics for raw SNR values sampled within a sphere."""

    mean: float
    median: float
    min: float
    max: float
    std: float
    n_points: int


@dataclass
class PerStructureSiteSNR:
    """
    SNR sphere statistics for one (water site, structure) pair.

    snr_site_radius: stats for a sphere of radius WaterSite.radius
    snr_1_5:         stats for a fixed 1.5 Å sphere
    snr_2_0:         stats for a fixed 2.0 Å sphere

    Any field is None when the sphere returned zero grid points (map boundary
    or zero-radius fallback edge case).
    """

    site_id: int
    stem: str
    has_water_modelled: bool
    snr_site_radius: Optional[SNRStats]
    snr_1_5: Optional[SNRStats]
    snr_2_0: Optional[SNRStats]


def _adaptive_max_spacing(radius: float, min_voxels: int, hard_max: float) -> float:
    """
    Compute the grid spacing required to guarantee at least min_voxels
    interpolated points inside a sphere of the given radius.

    Mirrors MUSE's oversampling strategy: the spacing fed to
    enumerate_grid_points_in_sphere controls how aggressively the native
    grid is oversampled per axis (factor = ceil(native_spacing / max_spacing)).

    Derivation:
        N ≈ V_sphere / V_voxel = (4/3 π r³) / d³
        d_required = ((4/3 π r³) / N_min)^(1/3)

    The returned value is clamped to hard_max so that we never use coarser
    spacing than the MUSE default (0.7 Å) even for large spheres.

    Args:
        radius: Sphere radius in Å.
        min_voxels: Target minimum voxel count.
        hard_max: Upper bound on spacing (Å); mirrors WaterSiteConfig.grid_max_spacing.

    Returns:
        max_spacing value to pass to enumerate_grid_points_in_sphere.
    """
    if radius <= 0.0 or min_voxels <= 0:
        return hard_max
    volume = (4.0 / 3.0) * math.pi * radius ** 3
    d_required = (volume / min_voxels) ** (1.0 / 3.0)
    return min(d_required, hard_max)


def _stats(values: np.ndarray) -> Optional[SNRStats]:
    if len(values) == 0:
        return None
    return SNRStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        std=float(np.std(values)),
        n_points=int(len(values)),
    )


def _empty_snr_row(site: WaterSite, stem: str, has_water: bool) -> PerStructureSiteSNR:
    return PerStructureSiteSNR(
        site_id=site.site_id,
        stem=stem,
        has_water_modelled=has_water,
        snr_site_radius=None,
        snr_1_5=None,
        snr_2_0=None,
    )


def extract_all_site_snr_for_structure(
    sites: List[WaterSite],
    paths: dict,
    transform: SuperpositionResult,
    snr_map_path: Optional[str],
    config: WaterSiteConfig,
) -> List[PerStructureSiteSNR]:
    """
    Load the SNR map for one structure once and extract sphere statistics for
    every water site.

    For each site the centroid is transformed from the reference frame back
    to this structure's coordinate frame before interpolation. Three sphere
    sizes are sampled:
        - site.radius   (computed per-site, floored at min_site_radius)
        - 1.5 Å         (fixed)
        - 2.0 Å         (fixed)

    Raw SNR values are used directly (no normalisation — these are already
    SNR maps). A None stats record is stored when a sphere returns zero grid
    points, distinguishing "no signal" from "no data".

    The grid spacing used for interpolation is chosen adaptively per sphere:
    it is the minimum of config.grid_max_spacing and the spacing required to
    guarantee at least config.min_voxels_per_sphere grid points inside the
    sphere (see _adaptive_max_spacing). This mirrors the MUSE oversampling
    strategy and ensures that even the smallest water site spheres have
    sufficient voxel count for reliable aggregation statistics.

    Args:
        sites: All water sites (reference frame centroids).
        paths: Experiment paths dict for this structure.
        transform: SuperpositionResult mapping mobile → reference (used
            in reverse here: reference → structure frame).
        snr_map_path: Pre-resolved path to the SNR CCP4 map, or None.
        config: WaterSiteConfig with grid interpolation parameters.

    Returns:
        One PerStructureSiteSNR per site, in the same order as sites.
    """
    stem = paths["stem"]
    has_water = {site.site_id: stem in site.member_stems for site in sites}

    if snr_map_path is None:
        return [_empty_snr_row(s, stem, has_water[s.site_id]) for s in sites]

    try:
        grid = load_map(snr_map_path)
    except Exception:
        return [_empty_snr_row(s, stem, has_water[s.site_id]) for s in sites]

    results: List[PerStructureSiteSNR] = []
    for site in sites:
        centroid_local = transform_to_structure(site.centroid, transform)
        center = gemmi.Position(*centroid_local.tolist())

        radii: List[Tuple[str, float]] = [
            ("site_r", site.radius),
            ("r1_5", 1.5),
            ("r2_0", 2.0),
        ]
        stats_by_label: Dict[str, Optional[SNRStats]] = {}
        for label, radius in radii:
            spacing = _adaptive_max_spacing(
                radius,
                config.min_voxels_per_sphere,
                config.grid_max_spacing,
            )
            _, values = enumerate_grid_points_in_sphere(
                grid,
                center,
                radius,
                max_spacing=spacing,
                interpolation_order=config.grid_interpolation_order,
            )
            stats_by_label[label] = _stats(values)

        results.append(
            PerStructureSiteSNR(
                site_id=site.site_id,
                stem=stem,
                has_water_modelled=has_water[site.site_id],
                snr_site_radius=stats_by_label["site_r"],
                snr_1_5=stats_by_label["r1_5"],
                snr_2_0=stats_by_label["r2_0"],
            )
        )

    return results
