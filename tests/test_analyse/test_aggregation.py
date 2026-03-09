from unittest.mock import MagicMock

from analyse.muse.aggregation import (
    aggregate_by_residue,
    compute_musem,
    compute_opia,
)
from analyse.muse.config import AggregationConfig


DEFAULT_CFG = AggregationConfig()


def test_compute_musem_empty():
    assert compute_musem([], DEFAULT_CFG) == 0.0


def test_compute_musem_single_atom():
    score = compute_musem([0.8], DEFAULT_CFG)
    assert 0.0 <= score <= 1.2


def test_compute_musem_all_perfect():
    score = compute_musem([1.0, 1.0, 1.0], DEFAULT_CFG)
    assert abs(score - 1.0) < 0.05


def test_compute_musem_soft_min_pulls_toward_worst():
    # With exponent -2, MUSEm is dominated by the lowest score
    scores_mixed = [1.0, 1.0, 0.1]
    scores_uniform = [0.7, 0.7, 0.7]
    musem_mixed = compute_musem(scores_mixed, DEFAULT_CFG)
    musem_uniform = compute_musem(scores_uniform, DEFAULT_CFG)
    # musem_mixed should be pulled below the arithmetic mean (0.7)
    assert musem_mixed < musem_uniform


def test_compute_musem_non_negative():
    for scores in [[0.0], [0.0, 0.0], [0.1, 0.05, 0.0]]:
        assert compute_musem(scores, DEFAULT_CFG) >= 0.0


def _atom(atom_id, score):
    a = MagicMock()
    a.atom_id = atom_id
    a.score = score
    return a


def test_compute_opia_empty():
    assert compute_opia([], {}) == 0.0


def test_compute_opia_all_poor():
    atoms = [_atom((0, 0, i), 0.3) for i in range(5)]
    assert compute_opia(atoms, {}, threshold=0.8) == 0.0


def test_compute_opia_all_well_supported_bonded():
    # Chain: 0→1→2→3 all bonded, all score ≥ 0.8
    atoms = [_atom((0, 0, i), 0.9) for i in range(4)]
    bonds = {
        (0, 0, 0): [(0, 0, 1)],
        (0, 0, 1): [(0, 0, 0), (0, 0, 2)],
        (0, 0, 2): [(0, 0, 1), (0, 0, 3)],
        (0, 0, 3): [(0, 0, 2)],
    }
    opia = compute_opia(atoms, bonds, threshold=0.8)
    assert abs(opia - 1.0) < 1e-9


def test_compute_opia_isolated_atoms_excluded():
    # Two isolated atoms with high score — no bonds → component size 1 → excluded
    atoms = [_atom((0, 0, 0), 0.9), _atom((0, 0, 1), 0.9)]
    opia = compute_opia(atoms, {}, threshold=0.8)
    assert opia == 0.0


def test_compute_opia_partial_support():
    # 4 atoms; 2 bonded + well-supported (included), 2 poor (excluded)
    atoms = [
        _atom((0, 0, 0), 0.9),
        _atom((0, 0, 1), 0.9),
        _atom((0, 0, 2), 0.2),
        _atom((0, 0, 3), 0.2),
    ]
    bonds = {(0, 0, 0): [(0, 0, 1)], (0, 0, 1): [(0, 0, 0)]}
    opia = compute_opia(atoms, bonds, threshold=0.8)
    assert abs(opia - 0.5) < 1e-9


def _atom_score(chain, res_name, seq_id, score, ins=""):
    a = MagicMock()
    a.chain_id = chain
    a.residue_name = res_name
    a.residue_seq_id = seq_id
    a.insertion_code = ins
    a.score = score
    a.has_clash = False
    a.has_missing_density = False
    a.has_unaccounted_density = False
    return a


def test_aggregate_by_residue_groups_correctly():
    atoms = [
        _atom_score("A", "ALA", 10, 0.8),
        _atom_score("A", "ALA", 10, 0.6),
        _atom_score("A", "GLY", 11, 0.9),
    ]
    residues = aggregate_by_residue(atoms, DEFAULT_CFG)
    assert len(residues) == 2


def test_aggregate_by_residue_sorted_by_chain_and_seq():
    atoms = [
        _atom_score("A", "ALA", 20, 0.7),
        _atom_score("A", "GLY", 5, 0.5),
        _atom_score("B", "VAL", 1, 0.9),
    ]
    residues = aggregate_by_residue(atoms, DEFAULT_CFG)
    assert residues[0].chain_id == "A" and residues[0].residue_seq_id == 5
    assert residues[1].chain_id == "A" and residues[1].residue_seq_id == 20
    assert residues[2].chain_id == "B"


def test_aggregate_by_residue_min_max_correct():
    atoms = [
        _atom_score("A", "ALA", 1, 0.2),
        _atom_score("A", "ALA", 1, 0.8),
        _atom_score("A", "ALA", 1, 0.5),
    ]
    residues = aggregate_by_residue(atoms, DEFAULT_CFG)
    assert len(residues) == 1
    r = residues[0]
    assert abs(r.min_atom_score - 0.2) < 1e-6
    assert abs(r.max_atom_score - 0.8) < 1e-6


def test_aggregate_by_residue_n_atoms():
    atoms = [_atom_score("A", "ALA", 1, 0.7) for _ in range(5)]
    residues = aggregate_by_residue(atoms, DEFAULT_CFG)
    assert residues[0].n_atoms == 5
