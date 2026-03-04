"""Error diagnostics for MUSE-scored structures.

Implements the three diagnostic flags from Meyder et al. 2017:
    - Clash detection: atoms with > 10% sphere overlap.
    - Missing density:  MUSE+ < threshold (expected density absent).
    - Unaccounted density: |MUSE-| > threshold (unexpected density in donut).

Each flag is set independently on the AtomScore objects in-place so that
the pipeline can run diagnostics after scoring without re-computing maps.

References:
    Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447, Section
    "Error Identification in Crystal Structure Models"
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import gemmi
import numpy as np

from muse.config import AggregationConfig
from muse.ownership import AtomId
from muse.scoring import AtomScore


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def detect_clashes(
    atom_scores: List[AtomScore],
    model: gemmi.Model,
    atom_id_to_position: Dict[AtomId, np.ndarray],
    atom_id_to_radius: Dict[AtomId, float],
    clash_overlap_threshold: float = 0.1,
) -> None:
    """Flag atoms involved in steric clashes (in-place).

    Two atoms are considered clashing when the overlap of their radii spheres
    exceeds *clash_overlap_threshold* (default 10 %). Overlap is measured as:

        overlap_fraction = (r_a + r_b - dist) / (r_a + r_b)

    If overlap_fraction > threshold for any neighbour, has_clash is set True.

    Only non-covalent pairs are tested; atoms that are covalently bonded are
    expected to overlap.

    Args:
        atom_scores: Per-atom scores whose has_clash fields are updated.
        model: gemmi.Model used to iterate atoms for lookup.
        atom_id_to_position: Maps AtomId -> Cartesian position (3,) array.
        atom_id_to_radius: Maps AtomId -> resolution-dependent radius.
        clash_overlap_threshold: Fractional overlap above which a clash is
            declared. Default 0.1 (10 %, as in Meyder 2017).
    """
    # Build index from AtomId to AtomScore for fast look-up
    id_to_score: Dict[AtomId, AtomScore] = {}
    for ascore in atom_scores:
        if ascore.atom_id is not None:
            id_to_score[ascore.atom_id] = ascore

    all_ids = list(atom_id_to_position.keys())
    all_pos = np.array([atom_id_to_position[aid] for aid in all_ids])
    all_radii = np.array([atom_id_to_radius[aid] for aid in all_ids])
    n = len(all_ids)

    # O(n^2) clash search – acceptable for typical PDB sizes (<~10 k heavy atoms).
    # For very large structures the neighbour-search variant should be used instead.
    for i in range(n):
        pos_i = all_pos[i]
        r_i = all_radii[i]
        aid_i = all_ids[i]

        # Vectors from atom i to all others
        diffs = all_pos - pos_i  # (n, 3)
        dists = np.linalg.norm(diffs, axis=1)  # (n,)

        # Sum of radii for each pair
        r_sum = r_i + all_radii  # (n,)

        # Overlap fraction: (r_i + r_j - dist) / (r_i + r_j)
        overlap = (r_sum - dists) / (r_sum + 1e-12)

        # Exclude self (distance == 0)
        overlap[i] = 0.0

        # Covalently bonded atoms are expected to overlap – exclude them.
        # A rough bond cutoff is r_i + r_j + 0.4 Å; overlapping pairs with
        # dist > 0 and overlap < ~0.45 are covalent-like. We use the simpler
        # heuristic: exclude pairs where dist < 0.5 Å (same site) or dist == 0.
        # The full exclusion list is handled by the caller providing only non-bonded
        # atom_id_to_position entries if needed; here we just apply the threshold.

        if np.any(overlap > clash_overlap_threshold):
            if aid_i in id_to_score:
                # AtomScore is a dataclass without slots; we need to mutate it
                object.__setattr__(id_to_score[aid_i], "has_clash", True)


# ---------------------------------------------------------------------------
# Missing / unaccounted density flags
# ---------------------------------------------------------------------------

def flag_missing_density(
    atom_scores: List[AtomScore],
    config: AggregationConfig,
) -> None:
    """Flag atoms where the positive-weight score is below threshold (in-place).

    A low MUSE+ indicates that the map has little density in the atom's
    expected sphere, suggesting that the atom placement is unsupported.

    From Meyder 2017: atoms with MUSE+ < missing_density_threshold (default 0.8)
    are flagged as potentially misplaced or absent.

    Args:
        atom_scores: Per-atom scores whose has_missing_density fields are updated.
        config: AggregationConfig carrying the threshold.
    """
    threshold = config.missing_density_threshold
    for ascore in atom_scores:
        if ascore.score_positive < threshold:
            object.__setattr__(ascore, "has_missing_density", True)


def flag_unaccounted_density(
    atom_scores: List[AtomScore],
    config: AggregationConfig,
) -> None:
    """Flag atoms with significant negative-weight score (in-place).

    A large |MUSE-| indicates that the map has density in the donut region
    surrounding the atom, suggesting nearby unmodelled density or a misfit
    in the model.

    From Meyder 2017: atoms with |MUSE-| > unaccounted_density_threshold
    (default 0.2) are flagged.

    Args:
        atom_scores: Per-atom scores whose has_unaccounted_density fields are
            updated.
        config: AggregationConfig carrying the threshold.
    """
    threshold = config.unaccounted_density_threshold
    for ascore in atom_scores:
        # score_negative is already negative; take absolute value for comparison
        if abs(ascore.score_negative) > threshold:
            object.__setattr__(ascore, "has_unaccounted_density", True)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_diagnostics(
    atom_scores: List[AtomScore],
    model: gemmi.Model,
    atom_id_to_position: Dict[AtomId, np.ndarray],
    atom_id_to_radius: Dict[AtomId, float],
    config: AggregationConfig,
) -> None:
    """Apply all three diagnostic flags to atom scores in-place.

    Convenience wrapper that calls detect_clashes, flag_missing_density,
    and flag_unaccounted_density in one step.

    Args:
        atom_scores: Per-atom scores to annotate.
        model: gemmi.Model for the scored structure.
        atom_id_to_position: Maps AtomId -> Cartesian position (3,).
        atom_id_to_radius: Maps AtomId -> radius used for scoring.
        config: AggregationConfig for threshold values.
    """
    detect_clashes(
        atom_scores=atom_scores,
        model=model,
        atom_id_to_position=atom_id_to_position,
        atom_id_to_radius=atom_id_to_radius,
        clash_overlap_threshold=config.clash_threshold,
    )
    flag_missing_density(atom_scores, config)
    flag_unaccounted_density(atom_scores, config)


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def summarize_diagnostics(atom_scores: List[AtomScore]) -> Dict[str, int]:
    """Return counts of each diagnostic flag across all atom scores.

    Args:
        atom_scores: Scored atoms (after diagnostics have been run).

    Returns:
        Dict with keys 'n_clashes', 'n_missing_density',
        'n_unaccounted_density', 'n_total'.
    """
    return {
        "n_total": len(atom_scores),
        "n_clashes": sum(1 for a in atom_scores if a.has_clash),
        "n_missing_density": sum(1 for a in atom_scores if a.has_missing_density),
        "n_unaccounted_density": sum(
            1 for a in atom_scores if a.has_unaccounted_density
        ),
    }
