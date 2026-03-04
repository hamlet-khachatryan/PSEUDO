"""Core per-atom MUSE score computation.

Implements the EDIA formula from Meyder et al. 2017 (eq 2) for protein/ligand
atoms and the simpler variant from Nittinger et al. 2015 (eq 1) for water
molecules. Also defines result dataclasses for per-atom and per-residue scores.

References:
    Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447, eq 2
    Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783, eq 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np

from muse.config import MUSEConfig
from muse.grid_utils import compute_distances, enumerate_grid_points_in_sphere
from muse.ownership import AtomId, compute_ownership_vectorized
from muse.weighting import (
    gaussian_weight_vectorized,
    parabolic_weight_vectorized,
)


@dataclass
class AtomScore:
    """Per-atom MUSE score and diagnostics.

    Attributes:
        chain_id: Chain identifier string.
        residue_name: Residue name (e.g., 'ALA', 'HOH').
        residue_seq_id: Residue sequence number.
        insertion_code: Residue insertion code (may be empty).
        atom_name: Atom name (e.g., 'CA', 'O').
        element: Element symbol.
        score: The MUSE score for this atom.
        score_positive: Contribution from positive-weight region only.
        score_negative: Contribution from negative-weight region only.
        is_water: Whether this atom was scored with water-specific logic.
        has_clash: Whether a steric clash was detected.
        has_unaccounted_density: Whether unaccounted density was flagged.
        has_missing_density: Whether missing density was flagged.
        radius_used: The resolution-dependent radius used for this atom.
        n_grid_points: Number of grid points in this atom's sphere.
        atom_id: Internal atom identifier tuple.
    """

    chain_id: str
    residue_name: str
    residue_seq_id: int
    insertion_code: str
    atom_name: str
    element: str
    score: float
    score_positive: float
    score_negative: float
    is_water: bool
    has_clash: bool = False
    has_unaccounted_density: bool = False
    has_missing_density: bool = False
    radius_used: float = 0.0
    n_grid_points: int = 0
    atom_id: Optional[AtomId] = None


@dataclass
class ResidueScore:
    """Aggregated score for a residue.

    Attributes:
        chain_id: Chain identifier.
        residue_name: Residue name.
        residue_seq_id: Residue sequence number.
        insertion_code: Residue insertion code.
        musem_score: Power-mean aggregated score over constituent atoms.
        min_atom_score: Minimum atomic MUSE score in this residue.
        median_atom_score: Median atomic MUSE score in this residue.
        max_atom_score: Maximum atomic MUSE score in this residue.
        atom_scores: Individual atom scores within this residue.
        n_atoms: Number of scored heavy atoms.
    """

    chain_id: str
    residue_name: str
    residue_seq_id: int
    insertion_code: str
    musem_score: float
    min_atom_score: float
    median_atom_score: float
    max_atom_score: float
    atom_scores: List[AtomScore] = field(default_factory=list)
    n_atoms: int = 0


@dataclass
class MUSEResult:
    """Complete result of a MUSE calculation.

    Attributes:
        atom_scores: All per-atom scores.
        residue_scores: Per-residue aggregated scores.
        opia: Overall OPIA value (fraction of well-supported atoms).
        global_mean: Map mean used for normalization.
        global_sigma: Map sigma used for normalization.
        config: The configuration used for this run.
    """

    atom_scores: List[AtomScore]
    residue_scores: List[ResidueScore]
    opia: float
    global_mean: float
    global_sigma: float
    config: MUSEConfig


def normalize_density_values(
    values: np.ndarray,
    mean: float,
    sigma: float,
    config: MUSEConfig,
) -> np.ndarray:
    """Apply z(p) normalization and optional truncation to map values.

    From Meyder 2017:
        z(p) = 0                          if (rho - mu)/sigma < 0
        z(p) = (rho - mu)/sigma           if 0 <= (rho - mu)/sigma <= zeta
        z(p) = zeta                       if (rho - mu)/sigma > zeta

    When normalization is disabled (for probability/SNR maps), raw values
    are optionally truncated at [0, zeta].

    Args:
        values: 1D array of raw map values.
        mean: Global map mean (mu).
        sigma: Global map standard deviation.
        config: Full MUSE configuration.

    Returns:
        1D array of processed density scores.
    """
    if config.map_normalization.normalize:
        # Standard EDIA normalization: z = (rho - mu) / sigma
        z = (values - mean) / sigma
    else:
        # For probability/SNR maps: use raw values
        z = values.copy()

    # Floor at zero
    z = np.maximum(z, 0.0)

    # Optional upper truncation at zeta
    if config.density_score.use_truncation:
        z = np.minimum(z, config.density_score.zeta)

    return z


def score_protein_atom(
    atom_pos: np.ndarray,
    atom_radius: float,
    atom_id: AtomId,
    grid: gemmi.FloatGrid,
    mean: float,
    sigma: float,
    covalent_neighbors: Dict[AtomId, List[AtomId]],
    nearby_atoms: List[Tuple[AtomId, np.ndarray, float]],
    config: MUSEConfig,
) -> Tuple[float, float, float, int]:
    """Compute the full MUSE score for a non-water atom.

    Implements equation 2 from Meyder 2017:
        MUSE(a) = sum[w(p,a) * o(p,a) * z(p)] / sum[w(p,a)] for w > 0

    Also computes the positive-weight and negative-weight sub-scores
    for diagnostic purposes.

    Args:
        atom_pos: Atom position as (3,) array.
        atom_radius: Resolution-dependent radius.
        atom_id: AtomId tuple for this atom.
        grid: The CCP4 map grid.
        mean: Global map mean.
        sigma: Global map sigma.
        covalent_neighbors: Bond adjacency dict.
        nearby_atoms: List of (AtomId, position(3,), radius) for atoms
            near this atom.
        config: Full MUSE configuration.

    Returns:
        Tuple of (score, score_positive, score_negative, n_grid_points).
    """
    center = gemmi.Position(*atom_pos.tolist())
    sphere_radius = 2.0 * atom_radius

    # Enumerate grid points in the sphere of interest (2 * r)
    positions, values = enumerate_grid_points_in_sphere(
        grid, center, sphere_radius,
        max_spacing=config.grid.max_spacing_angstrom,
        interpolation_order=config.grid.interpolation_order,
    )

    n_points = len(values)
    if n_points == 0:
        return 0.0, 0.0, 0.0, 0

    # Compute distances from grid points to scored atom
    distances = compute_distances(positions, atom_pos)

    # Compute weights (3-parabola)
    weights = parabolic_weight_vectorized(distances, atom_radius, config.weighting)

    # Compute ownership
    ownership = compute_ownership_vectorized(
        distances, positions, atom_id, atom_radius,
        nearby_atoms, covalent_neighbors,
    )

    # Normalize density values
    z = normalize_density_values(values, mean, sigma, config)

    # Compute the score components
    # MUSE(a) = sum[w * o * z] / sum[|w|] over positive-weight points (eq 2)
    # But we compute over ALL grid points in sphere of interest,
    # with positive w contributing positively and negative w contributing as penalty

    # Product w * o * z
    product = weights * ownership * z

    # Positive weight region
    pos_mask = weights > 0.0
    neg_mask = weights < 0.0

    sum_pos_weights = np.sum(np.abs(weights[pos_mask])) if np.any(pos_mask) else 0.0

    if sum_pos_weights == 0.0:
        return 0.0, 0.0, 0.0, n_points

    # Total score: sum of all w*o*z divided by sum of positive |w|
    score = np.sum(product) / sum_pos_weights

    # Sub-scores for diagnostics
    score_positive = np.sum(product[pos_mask]) / sum_pos_weights if np.any(pos_mask) else 0.0
    score_negative = np.sum(product[neg_mask]) / sum_pos_weights if np.any(neg_mask) else 0.0

    return float(score), float(score_positive), float(score_negative), n_points


def score_water_atom(
    atom_pos: np.ndarray,
    grid: gemmi.FloatGrid,
    mean: float,
    sigma: float,
    covalent_radius: float,
    vdw_radius: float,
    config: MUSEConfig,
) -> Tuple[float, float, float, int]:
    """Compute the water-specific MUSE score (Nittinger 2015 variant).

    Uses Gaussian + linear weighting within the van der Waals sphere.
    Only grid points with density above the sigma threshold contribute.
    No ownership logic is applied (water oxygens are isolated atoms).

    From Nittinger 2015, eq 1:
        MUSE(a) = (1/sum(omega)) * (1/sigma) * sum[omega * (f(p) - mu)]

    For probability/SNR maps (no normalization), raw values are used
    instead of (f(p) - mu)/sigma.

    Args:
        atom_pos: Water oxygen position as (3,) array.
        grid: The CCP4 map grid.
        mean: Global map mean.
        sigma: Global map sigma.
        covalent_radius: Covalent radius of oxygen (delta parameter).
        vdw_radius: Van der Waals radius of oxygen.
        config: Full MUSE configuration.

    Returns:
        Tuple of (score, score_positive, score_negative, n_grid_points).
        score_negative is always 0.0 for water (no negative weighting).
    """
    center = gemmi.Position(*atom_pos.tolist())

    # Enumerate grid points within vdW radius
    positions, values = enumerate_grid_points_in_sphere(
        grid, center, vdw_radius,
        max_spacing=config.grid.max_spacing_angstrom,
        interpolation_order=config.grid.interpolation_order,
    )

    n_points = len(values)
    if n_points == 0:
        return 0.0, 0.0, 0.0, 0

    # Compute distances
    distances = compute_distances(positions, atom_pos)

    # Compute Gaussian + linear weights
    weights = gaussian_weight_vectorized(distances, covalent_radius, vdw_radius)

    sum_weights = np.sum(weights)
    if sum_weights == 0.0:
        return 0.0, 0.0, 0.0, n_points

    # Apply sigma threshold and normalization
    if config.map_normalization.normalize:
        # Standard: only include values above sigma threshold, normalize by sigma
        threshold = mean + config.water_scoring.sigma_threshold * sigma
        above_threshold = values >= threshold
        density_contribution = np.where(above_threshold, values - mean, 0.0)
        score = np.sum(weights * density_contribution) / (sum_weights * sigma)
    else:
        # For probability/SNR maps: use raw values, optionally apply threshold
        if config.water_scoring.sigma_threshold > 0.0 and sigma > 0.0:
            threshold = mean + config.water_scoring.sigma_threshold * sigma
            density_contribution = np.where(values >= threshold, values, 0.0)
        else:
            density_contribution = np.maximum(values, 0.0)
        score = np.sum(weights * density_contribution) / sum_weights

    return float(score), float(score), 0.0, n_points
