"""MUSEm power-mean aggregation and OPIA calculation.

Implements the EDIAm aggregation formula from Meyder et al. 2017 (eq 7) and
the OPIA metric for quantifying the fraction of well-supported atoms.

The power mean with exponent -2 acts as a soft-minimum, making MUSEm sensitive
to individual poorly-scoring atoms within a residue or molecule.

References:
    Meyder et al. (2017) J. Chem. Inf. Model. 57, 2437-2447, eq 7
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from muse.config import AggregationConfig
from muse.scoring import AtomScore, ResidueScore


def compute_musem(
    scores: List[float],
    config: AggregationConfig,
) -> float:
    """Compute the MUSEm power-mean aggregate score.

    From Meyder 2017 eq 7:
        MUSEm(U) = (1/|U| * sum((MUSE(a) + shift)^exponent))^(1/exponent) - shift

    With default shift=0.1 and exponent=-2, this acts as a soft-minimum
    that is strongly influenced by any single low-scoring atom.

    Args:
        scores: List of per-atom MUSE scores for the fragment.
        config: Aggregation parameters (shift, exponent).

    Returns:
        The MUSEm aggregate score. Returns 0.0 for empty input.
    """
    if not scores:
        return 0.0

    n = len(scores)
    shift = config.ediam_shift
    exponent = config.ediam_exponent

    # Shifted scores
    shifted = np.array([s + shift for s in scores], dtype=np.float64)

    # Guard against zero or negative shifted values when using negative exponent
    shifted = np.maximum(shifted, 1e-10)

    # Power mean: (1/n * sum(x^p))^(1/p)
    powered = np.power(shifted, exponent)
    mean_powered = np.mean(powered)

    # Invert the exponent
    result = np.power(mean_powered, 1.0 / exponent)

    # Remove the shift
    result = float(result) - shift

    return max(result, 0.0)


def compute_opia(
    atom_scores: List[AtomScore],
    covalent_neighbors: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    threshold: float = 0.8,
) -> float:
    """Compute the OPIA metric (Overall Percentage of well-resolved Interconnected Atoms).

    OPIA is the fraction of heavy atoms that belong to connected substructures
    of at least two atoms where all atoms have MUSE >= threshold.

    From Meyder 2017: identifies connected components of well-supported
    atoms (MUSE >= 0.8) and sums their sizes, normalized by total atoms.

    Args:
        atom_scores: All per-atom scores.
        covalent_neighbors: Bond adjacency dict.
        threshold: MUSE threshold for "well-supported". Default 0.8.

    Returns:
        OPIA value in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not atom_scores:
        return 0.0

    # Identify well-supported atoms
    well_supported: Set[Tuple[int, int, int]] = set()
    for ascore in atom_scores:
        if ascore.atom_id is not None and ascore.score >= threshold:
            well_supported.add(ascore.atom_id)

    if not well_supported:
        return 0.0

    # Find connected components among well-supported atoms
    visited: Set[Tuple[int, int, int]] = set()
    total_in_good_components = 0

    for atom_id in well_supported:
        if atom_id in visited:
            continue

        # BFS to find the connected component
        component: Set[Tuple[int, int, int]] = set()
        queue = [atom_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            if current not in well_supported:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in covalent_neighbors.get(current, []):
                if neighbor not in visited and neighbor in well_supported:
                    queue.append(neighbor)

        # Only count components with at least 2 atoms
        if len(component) >= 2:
            total_in_good_components += len(component)

    total_atoms = len(atom_scores)
    return total_in_good_components / total_atoms if total_atoms > 0 else 0.0


def aggregate_by_residue(
    atom_scores: List[AtomScore],
    config: AggregationConfig,
) -> List[ResidueScore]:
    """Group atom scores by residue and compute MUSEm for each.

    Atoms are grouped by (chain_id, residue_name, residue_seq_id, insertion_code).

    Args:
        atom_scores: All per-atom scores.
        config: Aggregation parameters.

    Returns:
        List of ResidueScore, one per residue, sorted by chain and sequence.
    """
    # Group by residue key
    groups: Dict[Tuple[str, str, int, str], List[AtomScore]] = defaultdict(list)
    for ascore in atom_scores:
        key = (ascore.chain_id, ascore.residue_name,
               ascore.residue_seq_id, ascore.insertion_code)
        groups[key].append(ascore)

    residue_scores = []
    for (chain_id, res_name, seq_id, ins_code), atoms in groups.items():
        scores_list = [a.score for a in atoms]
        musem = compute_musem(scores_list, config)

        scores_arr = np.array(scores_list)
        residue_scores.append(ResidueScore(
            chain_id=chain_id,
            residue_name=res_name,
            residue_seq_id=seq_id,
            insertion_code=ins_code,
            musem_score=musem,
            min_atom_score=float(np.min(scores_arr)) if len(scores_arr) > 0 else 0.0,
            median_atom_score=float(np.median(scores_arr)) if len(scores_arr) > 0 else 0.0,
            max_atom_score=float(np.max(scores_arr)) if len(scores_arr) > 0 else 0.0,
            atom_scores=atoms,
            n_atoms=len(atoms),
        ))

    # Sort by chain, then by sequence number
    residue_scores.sort(key=lambda r: (r.chain_id, r.residue_seq_id))
    return residue_scores
