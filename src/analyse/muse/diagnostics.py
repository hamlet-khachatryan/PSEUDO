"""
References:
    - Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447
    - Nittinger et al. (2015) J. Chem. Inf. Model. 55, 771-783
"""

from __future__ import annotations

from typing import Dict, List

import gemmi
import numpy as np

from analyse.muse.config import AggregationConfig
from analyse.muse.ownership import AtomId
from analyse.muse.scoring import AtomScore


def detect_clashes(
    atom_scores: List[AtomScore],
    model: gemmi.Model,
    atom_id_to_position: Dict[AtomId, np.ndarray],
    atom_id_to_radius: Dict[AtomId, float],
    clash_overlap_threshold: float = 0.1,
) -> None:
    """
    Flag atoms involved in steric clashes

    Args:
        atom_scores: Per-atom scores whose has_clash fields are updated
        model: gemmi.Model used for atoms lookup
        atom_id_to_position: Maps AtomId -> position (3,) array
        atom_id_to_radius: Maps AtomId -> resolution-dependent radius
        clash_overlap_threshold: Fractional overlap
    """
    id_to_score: Dict[AtomId, AtomScore] = {}
    for ascore in atom_scores:
        if ascore.atom_id is not None:
            id_to_score[ascore.atom_id] = ascore

    all_ids = list(atom_id_to_position.keys())
    all_pos = np.array([atom_id_to_position[aid] for aid in all_ids])
    all_radii = np.array([atom_id_to_radius[aid] for aid in all_ids])
    n = len(all_ids)

    # O(n^2) clash search
    for i in range(n):
        pos_i = all_pos[i]
        r_i = all_radii[i]
        aid_i = all_ids[i]

        diffs = all_pos - pos_i
        dists = np.linalg.norm(diffs, axis=1)
        r_sum = r_i + all_radii

        overlap = (r_sum - dists) / (r_sum + 1e-12)
        overlap[i] = 0.0

        if np.any(overlap > clash_overlap_threshold):
            if aid_i in id_to_score:
                object.__setattr__(id_to_score[aid_i], "has_clash", True)


def flag_missing_density(
    atom_scores: List[AtomScore],
    config: AggregationConfig,
) -> None:
    """
    Flag atoms where the positive-weight score is below the threshold, suggesting that the atom placement is unsupported

    Args:
        atom_scores: Per-atom scores whose has_missing_density fields are updated
        config: AggregationConfig carrying the threshold
    """
    threshold = config.missing_density_threshold
    for ascore in atom_scores:
        if ascore.score_positive < threshold:
            object.__setattr__(ascore, "has_missing_density", True)


def flag_unaccounted_density(
    atom_scores: List[AtomScore],
    config: AggregationConfig,
) -> None:
    """
    Flag atoms with significant negative-weight score, suggesting nearby unmodelled density or a misfit

    Args:
        atom_scores: Per-atom scores whose has_unaccounted_density fields are
            updated.
        config: AggregationConfig carrying the threshold
    """
    threshold = config.unaccounted_density_threshold
    for ascore in atom_scores:
        if abs(ascore.score_negative) > threshold:
            object.__setattr__(ascore, "has_unaccounted_density", True)


def run_diagnostics(
    atom_scores: List[AtomScore],
    model: gemmi.Model,
    atom_id_to_position: Dict[AtomId, np.ndarray],
    atom_id_to_radius: Dict[AtomId, float],
    config: AggregationConfig,
) -> None:
    """
    Calls detect_clashes, flag_missing_density, and flag_unaccounted_density
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


def summarize_diagnostics(atom_scores: List[AtomScore]) -> Dict[str, int]:
    """
    Return counts of each diagnostic flag across all atom scores

    Args:
        atom_scores: Scored atoms
    Returns:
        Dict with keys 'n_clashes', 'n_missing_density', 'n_unaccounted_density', 'n_total'.
    """
    return {
        "n_total": len(atom_scores),
        "n_clashes": sum(1 for a in atom_scores if a.has_clash),
        "n_missing_density": sum(1 for a in atom_scores if a.has_missing_density),
        "n_unaccounted_density": sum(
            1 for a in atom_scores if a.has_unaccounted_density
        ),
    }
