"""Configuration dataclasses for MUSE scoring.

All paper-derived parameters are stored as frozen dataclasses with defaults
matching the original EDIA publications. Users construct a MUSEConfig with
overrides for their specific map type (probability, SNR, electron density).

References:
    - Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    - Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DensityScoreConfig:
    """Parameters for the density score z(p) truncation.

    Controls how raw map values are clamped before scoring.
    From Meyder 2017: z(p) is truncated at zeta (default 1.2 sigma).

    Attributes:
        zeta: Upper truncation threshold for normalized density values.
            Default 1.2 from Meyder 2017.
        use_truncation: If False, pass normalized values without upper
            truncation. Useful when the CCP4 map values should not be
            clipped (e.g. probability maps with meaningful high values).
    """

    zeta: float = 1.2
    use_truncation: bool = True


@dataclass(frozen=True)
class WeightingConfig:
    """Parameters for the 3-parabola weighting curve (Meyder 2017, SI Table S3).

    The curve assigns positive weights to grid points inside the electron
    density sphere s(a) and negative weights in the surrounding donut d(a).
    All distance parameters are expressed as fractions of the atom radius r.

    Each parabola has the form P(x) = m * (x - c*r)^2 + b, where the m values
    are divided by r^2 internally to make them resolution-independent.

    Attributes:
        p1_m: Curvature of P1. Default -1.0.
        p1_c_frac: Center of P1 as fraction of r. Default 0.0.
        p1_b: P1 value at center. Default 1.0.
        transition_1_frac: Fraction of r where P1 transitions to P2.
            Default 1.0822.
        p2_m: Curvature of P2. Default 5.1177.
        p2_c_frac: Center of P2 as fraction of r. Default 1.29366.
        p2_b: Minimum value of P2. Default -0.4.
        transition_2_frac: Fraction of r where P2 transitions to P3.
            Default 1.4043.
        p3_m: Curvature of P3. Default -0.9507.
        p3_c_frac: Center of P3 as fraction of r. Default 2.0.
        p3_b: P3 value at endpoint. Default 0.0.
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
    """Parameters for the water-specific scoring variant (Nittinger 2015).

    Water molecules (single oxygen, not covalently bonded) use a simpler
    Gaussian + linear weighting within the van der Waals radius.

    Attributes:
        sigma_threshold: Only map values >= sigma_threshold * sigma
            contribute to the score. Default 1.0 (1-sigma cutoff).
            Set to 0.0 to include all positive density.
        classification_threshold: MUSE threshold below which a water is
            classified as insufficiently resolved. Default 0.24
            (median - 1 std from Nittinger 2015).
    """

    sigma_threshold: float = 1.0
    classification_threshold: float = 0.24


@dataclass(frozen=True)
class GridConfig:
    """Parameters controlling grid oversampling and interpolation.

    From Meyder 2017 SI 2.1.2: the grid is oversampled to guarantee
    at least 27 grid points in the electron density radius sphere.

    Attributes:
        max_spacing_angstrom: Maximum allowed grid spacing in Angstroms
            before oversampling kicks in. Default 0.7 A.
        interpolation_order: Order for map value interpolation.
            1 = trilinear, 3 = tricubic. Default 3 (as in the paper).
    """

    max_spacing_angstrom: float = 0.7
    interpolation_order: int = 3


@dataclass(frozen=True)
class OwnershipConfig:
    """Parameters for grid point ownership among neighboring atoms.

    Ownership determines how density at a grid point is attributed when
    multiple atoms claim it. Covalently bonded atoms share density fully;
    non-bonded overlapping atoms share by inverse distance.

    Attributes:
        covalent_bond_tolerance: Distance tolerance (Angstroms) added to
            the sum of covalent radii when determining if two atoms are
            bonded. Default 0.4 A.
    """

    covalent_bond_tolerance: float = 0.4


@dataclass(frozen=True)
class AggregationConfig:
    """Parameters for MUSEm and OPIA aggregation (Meyder 2017, eq 7).

    MUSEm uses a power mean: (1/|U| * sum((s + shift)^exponent))^(1/exponent) - shift.
    With default exponent=-2, this behaves as a soft-minimum that is sensitive
    to individual poorly-scoring atoms.

    Attributes:
        ediam_shift: Additive shift to prevent division by zero in the
            power mean. Default 0.1.
        ediam_exponent: Power mean exponent. Default -2.
        opia_threshold: MUSE score threshold for an atom to be considered
            well-supported in OPIA calculation. Default 0.8.
        clash_threshold: Overlap fraction above which a steric clash is
            flagged between two atoms. Default 0.1.
        unaccounted_density_threshold: Threshold for the negative-weight
            MUSE component (MUSE-). Default 0.2.
        missing_density_threshold: Threshold below which the positive-weight
            MUSE component (MUSE+) indicates missing support. Default 0.8.
    """

    ediam_shift: float = 0.1
    ediam_exponent: float = -2.0
    opia_threshold: float = 0.8
    clash_threshold: float = 0.1
    unaccounted_density_threshold: float = 0.2
    missing_density_threshold: float = 0.8


@dataclass(frozen=True)
class MapNormalizationConfig:
    """Controls how map values are normalized before scoring.

    For standard electron density maps, EDIA normalizes values as
    z = (rho - mu) / sigma. For probability or SNR maps the values are
    already meaningful, so normalization should typically be skipped.

    Attributes:
        normalize: Whether to apply (value - mu) / sigma normalization.
            Default False (suitable for probability/SNR maps). Set True
            for standard electron density maps.
        global_mean_override: If set, use this as the map mean instead
            of computing it. Default None (auto-compute).
        global_sigma_override: If set, use this as the map standard
            deviation instead of computing it. Default None (auto-compute).
    """

    normalize: bool = False
    global_mean_override: Optional[float] = None
    global_sigma_override: Optional[float] = None


@dataclass(frozen=True)
class MUSEConfig:
    """Top-level configuration aggregating all sub-configs.

    Construct with defaults and override individual fields:

        cfg = MUSEConfig(
            density_score=DensityScoreConfig(zeta=1.5),
            map_normalization=MapNormalizationConfig(normalize=True),
        )

    Attributes:
        density_score: Truncation parameters for map values.
        weighting: Parabolic weighting curve parameters.
        water_scoring: Water-specific scoring parameters.
        grid: Grid oversampling parameters.
        ownership: Atom ownership parameters.
        aggregation: MUSEm aggregation parameters.
        map_normalization: Map value normalization parameters.
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
    """Return the default configuration with all paper-derived values.

    Returns:
        A MUSEConfig instance with normalization disabled (suitable for
        probability/SNR maps) and all other parameters matching the
        original EDIA publications.
    """
    return MUSEConfig()
