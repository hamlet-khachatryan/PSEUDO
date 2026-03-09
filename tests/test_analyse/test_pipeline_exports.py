from unittest.mock import MagicMock
import numpy as np

from analyse.muse.pipeline import export_summary

def _atom(score, clash=False, missing=False, unaccounted=False):
    a = MagicMock()
    a.score = score
    a.has_clash = clash
    a.has_missing_density = missing
    a.has_unaccounted_density = unaccounted
    return a


def _result(atom_scores, residue_scores=None):
    r = MagicMock()
    r.atom_scores = atom_scores
    r.residue_scores = residue_scores or []
    r.opia = 0.75
    r.global_mean = 1.2
    r.global_sigma = 0.8
    return r

def test_export_summary_keys():
    result = _result([_atom(0.8), _atom(0.5)])
    summary = export_summary(result)
    expected_keys = {
        "n_atoms", "n_residues", "opia",
        "mean_atom_score", "median_atom_score",
        "n_clashes", "n_missing_density", "n_unaccounted_density",
        "global_mean", "global_sigma",
    }
    assert expected_keys == set(summary.keys())


def test_export_summary_counts():
    atoms = [
        _atom(0.9),
        _atom(0.2, clash=True),
        _atom(0.1, missing=True),
        _atom(0.3, unaccounted=True),
    ]
    summary = export_summary(_result(atoms))
    assert summary["n_atoms"] == 4
    assert summary["n_clashes"] == 1
    assert summary["n_missing_density"] == 1
    assert summary["n_unaccounted_density"] == 1


def test_export_summary_mean_score():
    atoms = [_atom(0.4), _atom(0.6), _atom(0.8)]
    summary = export_summary(_result(atoms))
    assert abs(summary["mean_atom_score"] - np.mean([0.4, 0.6, 0.8])) < 1e-6


def test_export_summary_empty_atoms():
    summary = export_summary(_result([]))
    assert summary["n_atoms"] == 0
    assert summary["mean_atom_score"] == 0.0
    assert summary["median_atom_score"] == 0.0


def test_export_summary_opia_passthrough():
    result = _result([_atom(0.9)])
    result.opia = 0.83
    summary = export_summary(result)
    assert abs(summary["opia"] - 0.83) < 1e-9
