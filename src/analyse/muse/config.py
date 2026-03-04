"""
References:
    - Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    - Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DensityScoreConfig:
    """
    Parameters for the density score z(p) truncation

    Attributes:
        zeta: Upper truncation threshold for normalized density values. Default 1.2.
        use_truncation: If False, pass normalized values without upper
            truncation. Useful when the CCP4 map values should not be
            clipped (e.g. SNR maps with meaningful high values).
    """

    zeta: float = 1.2
    use_truncation: bool = True


@dataclass(frozen=True)
class WeightingConfig:
    """
    Parameters for the 3-parabola weighting curve

    The curve assigns positive weights to grid points inside the electron
    density sphere s(a) and negative weights in the surrounding donut d(a).

    Attributes:
        p1_m: Curvature of P1
        p1_c_frac: Center of P1 as fraction of r
        p1_b: P1 value at center
        transition_1_frac: Fraction of r where P1 transitions to P2
        p2_m: Curvature of P2
        p2_c_frac: Center of P2 as fraction of r
        p2_b: Minimum value of P2
        transition_2_frac: Fraction of r where P2 transitions to P3
        p3_m: Curvature of P3
        p3_c_frac: Center of P3 as fraction of r
        p3_b: P3 value at endpoint
    """

    p1_m: float = -1.0
    p1_c_frac: float = 0.0
    p1_b: float = 1.0
    transition_1_frac: float = 1.0822
    p2_m: float = 5.1177
    p2_c_frac: float = 1.29366
    p2_b: float = -0.4
    transition_2_frac: float = 1.4043
    p3_m: float = -0.9507
    p3_c_frac: float = 2.0
    p3_b: float = 0.0


@dataclass(frozen=True)
class WaterScoringConfig:
    """
    Parameters for the water-specific scoring  (Nittinger 2015)
    Gaussian + linear weighting within the van der Waals radius

    Attributes:
        sigma_threshold: Only map values >= sigma_threshold * sigma
            contribute to the score. Default 1.0 (1-sigma cutoff)
        classification_threshold: MUSE threshold below which a water is
            classified as insufficiently resolved. Default 0.24
    """

    sigma_threshold: float = 1.0
    classification_threshold: float = 0.24


@dataclass(frozen=True)
class GridConfig:
    """
    Grid oversampling and interpolation params

    Attributes:
        max_spacing_angstrom: Maximum grid spacing before oversampling
        interpolation_order: Order for map value interpolation. 1 = trilinear, 3 = tricubic
    """

    max_spacing_angstrom: float = 0.7
    interpolation_order: int = 3


@dataclass(frozen=True)
class OwnershipConfig:
    """
    Parameters for grid point ownership

    Attributes:
        covalent_bond_tolerance: Distance tolerance added to
            the sum of covalent radii
    """

    covalent_bond_tolerance: float = 0.4


@dataclass(frozen=True)
class AggregationConfig:
    """
    Parameters for MUSEm and OPIA aggregation

    Attributes:
        ediam_shift: Additive shift to prevent division by zero in the power mean
        ediam_exponent: Power mean exponent
        opia_threshold: MUSE score threshold for an atom to be considered well-supported
        clash_threshold: Overlap fraction above which a steric clash
        unaccounted_density_threshold: Threshold for the negative-weight MUSE component (MUSE-)
        missing_density_threshold: Threshold below which the positive-weight MUSE component (MUSE+)
    """

    ediam_shift: float = 0.1
    ediam_exponent: float = -2.0
    opia_threshold: float = 0.8
    clash_threshold: float = 0.1
    unaccounted_density_threshold: float = 0.2
    missing_density_threshold: float = 0.8


@dataclass(frozen=True)
class MapNormalizationConfig:
    """
    Controls how map values are normalized before scoring

    For standard electron density maps, EDIA normalizes values as z = (rho - mu) / sigma
    For probability or SNR maps the normalization should typically be skipped

    Attributes:
        normalize: Whether to apply normalization
        global_mean_override: If set, use this as the map mean instead of computing it
        global_sigma_override: If set, use this as the map standard deviation instead of computing it
    """

    normalize: bool = False
    global_mean_override: Optional[float] = None
    global_sigma_override: Optional[float] = None


@dataclass(frozen=True)
class MUSEConfig:
    """
    Top-level configuration aggregating all sub-configs
    Construct with defaults and override individual fields

    Attributes:
        density_score: Truncation parameters for map values
        weighting: Parabolic weighting curve parameters
        water_scoring: Water-specific scoring parameters
        grid: Grid oversampling parameters
        ownership: Atom ownership parameters
        aggregation: MUSEm aggregation parameters
        map_normalization: Map value normalization parameters
    """

    density_score: DensityScoreConfig = field(default_factory=DensityScoreConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    water_scoring: WaterScoringConfig = field(default_factory=WaterScoringConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    ownership: OwnershipConfig = field(default_factory=OwnershipConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    map_normalization: MapNormalizationConfig = field(
        default_factory=MapNormalizationConfig
    )


def default_config() -> MUSEConfig:
    """
    Return the default configuration with all paper-derived values
    """
    return MUSEConfig()
