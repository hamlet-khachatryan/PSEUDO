from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gemmi
import numpy as np


@dataclass
class SuperpositionResult:
    """
    Rigid-body transform that maps mobile coordinates to reference coordinates.

        ref_pos  ≈  rotation @ mob_pos + translation
        mob_pos  ≈  rotation.T @ (ref_pos - translation)

    Attributes:
        rotation: (3, 3) rotation matrix.
        translation: (3,) translation vector.
        rmsd: Cα RMSD after alignment in Å.
        n_ca_used: Number of common Cα pairs used for the superposition.
        is_identity: True when there were insufficient Cα pairs and the
            identity transform was used as a fallback.
    """

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    n_ca_used: int
    is_identity: bool


def _extract_ca_positions(
    structure: gemmi.Structure,
) -> Dict[Tuple[str, int], np.ndarray]:
    """Return CA positions keyed by (chain_name, residue_seq_id)."""
    cas: Dict[Tuple[str, int], np.ndarray] = {}
    model = structure[0]
    for chain in model:
        for residue in chain:
            ca = residue.find_atom("CA", "\0")
            if ca is None:
                continue
            key = (chain.name, residue.seqid.num)
            cas[key] = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=np.float64)
    return cas


def _kabsch(
    P: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Kabsch algorithm: find R, t minimising RMSD for the mapping Q → P.

    Args:
        P: (N, 3) reference positions.
        Q: (N, 3) mobile positions.

    Returns:
        R: (3, 3) rotation matrix  (P ≈ R @ Q + t)
        t: (3,) translation vector
        rmsd: root-mean-square deviation after alignment
    """
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    P_cent = P - p_c
    Q_cent = Q - q_c

    H = Q_cent.T @ P_cent
    U, _, Vt = np.linalg.svd(H)

    # Correct for improper rotation (reflection)
    det_sign = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, det_sign])
    R = Vt.T @ D @ U.T

    t = p_c - R @ q_c
    diff = (R @ Q.T).T + t - P
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    return R, t, rmsd


def compute_alignment(
    reference: gemmi.Structure,
    mobile: gemmi.Structure,
    min_overlap: int = 20,
) -> SuperpositionResult:
    """
    Compute a Cα-based rigid-body superposition of mobile onto reference.

    Falls back to the identity transform when the number of common Cα
    residues is below min_overlap.

    Args:
        reference: The fixed reference structure.
        mobile: The structure to be aligned onto reference.
        min_overlap: Minimum common Cα residue pairs required for reliable
            superposition.

    Returns:
        SuperpositionResult with rotation and translation mapping
        mobile coordinates to the reference frame.
    """
    ref_cas = _extract_ca_positions(reference)
    mob_cas = _extract_ca_positions(mobile)

    common = sorted(set(ref_cas) & set(mob_cas))

    if len(common) < min_overlap:
        return SuperpositionResult(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
            rmsd=0.0,
            n_ca_used=len(common),
            is_identity=True,
        )

    P = np.array([ref_cas[k] for k in common], dtype=np.float64)
    Q = np.array([mob_cas[k] for k in common], dtype=np.float64)

    R, t, rmsd = _kabsch(P, Q)
    return SuperpositionResult(
        rotation=R,
        translation=t,
        rmsd=rmsd,
        n_ca_used=len(common),
        is_identity=False,
    )


def transform_to_reference(pos: np.ndarray, t: SuperpositionResult) -> np.ndarray:
    """Apply mobile → reference transform: R @ pos + translation."""
    return t.rotation @ pos + t.translation


def transform_to_structure(pos: np.ndarray, t: SuperpositionResult) -> np.ndarray:
    """Apply reference → mobile (inverse) transform: R^T @ (pos - translation)."""
    return t.rotation.T @ (pos - t.translation)
