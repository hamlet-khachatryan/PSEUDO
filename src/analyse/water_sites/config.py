from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class WaterSiteConfig:
    """
    Configuration for water site consistency analysis across a screening.

    Attributes:
        clustering_eps: DBSCAN neighbourhood radius in Å. Waters within this
            distance are candidates for the same site.
        clustering_min_samples: Minimum cluster size. 1 means every isolated
            water forms its own site.
        min_site_radius: Floor radius (Å) for SNR sphere extraction. Applied
            when the computed site radius is smaller, e.g. for single-water
            sites. Defaults to the VdW radius of oxygen (1.52 Å).
        occupancy_search_radius: Radius (Å) used to find the nearest heavy
            atom when classifying what occupies a water site.
        fixed_snr_radii: Fixed sphere radii (Å) used for SNR extraction
            alongside the per-site computed radius.
        min_ca_overlap: Minimum common Cα residues required for a reliable
            superposition. Structures below this receive an identity transform.
        min_voxels_per_sphere: Minimum number of grid points that must fall
            inside a sphere for the aggregated statistics to be meaningful.
            The grid spacing is adaptively tightened per sphere to guarantee
            this count before falling back to None (no-data). A value of 10
            is a practical lower bound for median/std to be reliable.
        grid_max_spacing: Hard upper limit on grid spacing (Å) used for SNR
            interpolation, mirroring the MUSE oversampling parameter. The
            adaptive per-sphere spacing is always clamped to this value.
        grid_interpolation_order: Interpolation order passed to
            enumerate_grid_points_in_sphere. 1 = trilinear, 3 = tricubic.
        k_factor: K factor used to locate the per-experiment SNR map.
        map_cap: Map cap used to locate the per-experiment SNR map.
    """

    clustering_eps: float = 1.5
    clustering_min_samples: int = 1
    min_site_radius: float = 1.52
    occupancy_search_radius: float = 1.5
    fixed_snr_radii: List[float] = field(default_factory=lambda: [1.5, 2.0])
    min_ca_overlap: int = 20
    min_voxels_per_sphere: int = 10
    grid_max_spacing: float = 0.7
    grid_interpolation_order: int = 3
    k_factor: float = 1.0
    map_cap: Optional[int] = 50
