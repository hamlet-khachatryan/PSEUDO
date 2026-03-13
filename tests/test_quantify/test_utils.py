import pytest
from pathlib import Path

from quantify.utils import (
    find_experiments,
    infer_omission_mode,
    validate_experiment,
)

def _make_omission_map(mode: str):
    """Return a minimal omission-map dict for the given mode."""
    if mode == "amino_acids":
        return {
            "A|10|ALA|CA": [0, 2, 4],
            "A|10|ALA|CB": [0, 2, 4],
            "A|11|GLY|CA": [1, 3],
        }
    else:
        return {
            "A|10|ALA|CA": [0, 2, 4],
            "A|10|ALA|CB": [1, 3, 5],
        }


def test_infer_omission_mode_amino_acids():
    assert infer_omission_mode(_make_omission_map("amino_acids")) == "amino_acids"


def test_infer_omission_mode_atoms():
    assert infer_omission_mode(_make_omission_map("atoms")) == "atoms"


def test_infer_omission_mode_empty():
    assert infer_omission_mode({}) == "amino_acids"

def test_validate_experiment_valid(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "stem_updated.pdb").touch()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "stem_omission_map.json").touch()

    from quantify.utils import get_experiment_paths
    paths = get_experiment_paths(tmp_path, "stem")
    assert validate_experiment(paths) is True


def test_validate_experiment_missing_pdb(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "stem_omission_map.json").touch()
    (tmp_path / "processed").mkdir()

    from quantify.utils import get_experiment_paths
    paths = get_experiment_paths(tmp_path, "stem")
    assert validate_experiment(paths) is False

def _make_experiment(root: Path, stem: str):
    processed = root / "processed"
    processed.mkdir(parents=True)
    (processed / f"{stem}_updated.pdb").touch()
    meta = root / "metadata"
    meta.mkdir()
    (meta / f"{stem}_omission_map.json").write_text("{}")


def test_find_experiments_single(tmp_path):
    stem = tmp_path.name
    _make_experiment(tmp_path, stem)
    results = list(find_experiments(str(tmp_path)))
    assert len(results) == 1
    assert results[0]["stem"] == stem


def test_find_experiments_screening(tmp_path):
    for name in ["xtal_a", "xtal_b", "xtal_c"]:
        _make_experiment(tmp_path / name, name)
    results = list(find_experiments(str(tmp_path)))
    assert len(results) == 3
    stems = {r["stem"] for r in results}
    assert stems == {"xtal_a", "xtal_b", "xtal_c"}


def test_find_experiments_missing_path():
    with pytest.raises(FileNotFoundError):
        list(find_experiments("/nonexistent/path"))
