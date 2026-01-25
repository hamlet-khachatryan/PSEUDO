import pytest
from pathlib import Path
from src.debias import omission_sampler as sample

PDB_TEST = str(Path(__file__).parent.parent / "test_data/test_model.pdb")


@pytest.mark.skipif(not PDB_TEST, reason="No PDB files in test_data")
@pytest.mark.parametrize("pdb_file_param", [PDB_TEST])
def test_stochastic_sampler(pdb_file_param: str):
    ids = sample.extract_ids(pdb_file_param, mode="amino_acids")
    assert isinstance(ids, list)
    assert len(ids) > 0
    selectors = ids[:2]
    selection_string = ",".join([" ".join(map(str, i)) for i in selectors])
    selections = sample.stochastic_omission_sampler(
        pdb_file_param,
        omit_type="amino_acids",
        omit_fraction=0.3,
        n_iterations=2,
        always_omit=selection_string,
        seed=42,
    )

    assert isinstance(selections, list)
    assert len(selections) == 8
    for s in selections:
        for sel in selectors:
            assert sel in s


@pytest.mark.parametrize("pdb_file_param", [PDB_TEST])
def test_extract_atom_ids(pdb_file_param: str):
    atom_ids = sample.extract_ids(pdb_file_param, mode="atoms")
    assert isinstance(atom_ids, list)
    assert len(atom_ids) > 0

    for atom in atom_ids[:20]:
        assert isinstance(atom, tuple)
        assert len(atom) == 5


def test_invalid_structure_path():
    with pytest.raises(IOError):
        sample.stochastic_omission_sampler(
            "nonexistent.pdb", omit_type="amino_acids", omit_fraction=0.2
        )


@pytest.mark.parametrize("bad_frac", [-0.1, 0.0, 1.0, 1.5])
@pytest.mark.parametrize("pdb_file_param", [PDB_TEST])
def test_invalid_fraction(bad_frac: float, pdb_file_param: str):
    with pytest.raises(ValueError):
        sample._validate_sampler_inputs(pdb_file_param, bad_frac, None)


@pytest.mark.parametrize("pdb_file_param", [PDB_TEST])
def test_invalid_n_iterations(pdb_file_param: str):
    with pytest.raises(ValueError):
        sample._validate_sampler_inputs(pdb_file_param, 0.2, 0)


@pytest.mark.parametrize("pdb_file_param", [PDB_TEST])
def test_always_omit_with_missing_selectors(pdb_file_param: str):
    ids = sample.extract_ids(pdb_file_param, mode="amino_acids")
    assert ids, "PDB should yield amino-acid ids for this test"

    real_selector = ids[0]
    fake_selector_1 = ("Z", 99999, "FOO")
    fake_selector_2 = ("Z", 99998, "BAR")
    selectors = [real_selector, fake_selector_1, fake_selector_2]
    selection_string = ",".join([" ".join(map(str, i)) for i in selectors])

    selections = sample.stochastic_omission_sampler(
        pdb_file_param,
        omit_type="amino_acids",
        omit_fraction=0.3,
        n_iterations=1,
        always_omit=selection_string,
        seed=7,
    )
    assert isinstance(selections, list)
    assert len(selections) > 0

    for s in selections:
        assert real_selector in s
        assert fake_selector_1 not in s
        assert fake_selector_2 not in s
