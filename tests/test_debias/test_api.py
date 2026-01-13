import pytest

from debias.config import DebiasConfig
from debias.api import _config_validator


@pytest.fixture
def valid_manual_config(tmp_path):
    structure_path = tmp_path / "structure.pdb"
    reflections_path = tmp_path / "reflections.mtz"
    structure_path.touch()
    reflections_path.touch()
    return DebiasConfig(
        debias=type(
            "Debias",
            (),
            {
                "structure_path": structure_path,
                "reflections_path": reflections_path,
                "screening_path": None,
            },
        )
    )


@pytest.fixture
def valid_screening_config(tmp_path):
    screening_path = tmp_path / "screening.csv"
    screening_path.touch()
    return DebiasConfig(
        debias=type(
            "Debias",
            (),
            {
                "structure_path": None,
                "reflections_path": None,
                "screening_path": screening_path,
            },
        )
    )


@pytest.fixture
def invalid_config():
    return DebiasConfig(
        debias=type(
            "Debias",
            (),
            {"structure_path": None, "reflections_path": None, "screening_path": None},
        )
    )


@pytest.fixture
def missing_files_config(tmp_path):
    return DebiasConfig(
        debias=type(
            "Debias",
            (),
            {
                "structure_path": tmp_path / "missing_structure.pdb",
                "reflections_path": tmp_path / "missing_reflections.mtz",
                "screening_path": None,
            },
        )
    )


def test_valid_manual_config(valid_manual_config):
    try:
        _config_validator(valid_manual_config)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


def test_valid_screening_config(valid_screening_config):
    try:
        _config_validator(valid_screening_config)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


def test_invalid_config(invalid_config):
    with pytest.raises(
        ValueError, match="Invalid Configuration: Missing input data sources"
    ):
        _config_validator(invalid_config)


def test_missing_files(missing_files_config):
    with pytest.raises(FileNotFoundError, match="Structure file not found at:"):
        _config_validator(missing_files_config)
