import json
import pytest

from analyse.api import (
    _resolve_map_path,
    _resolve_model_path,
    _load_null_params,
    _infer_resolution,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _paths(tmp_path, stem="target"):
    """Minimal paths dict mirroring get_experiment_paths output."""
    return {
        "stem": stem,
        "root": tmp_path,
        "quantify_dir": tmp_path / "quantify_results",
        "processed_pdb": tmp_path / "processed" / f"{stem}_updated.pdb",
        "original_pdb": tmp_path / "processed" / f"{stem}_original.pdb",
        "metadata_dir": tmp_path / "metadata",
        "results_dir": tmp_path / "results",
    }


# ── _resolve_map_path ─────────────────────────────────────────────────────────

def test_resolve_map_path_missing_quantify_dir_raises(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="No quantify_results directory"):
        _resolve_map_path(paths, k_factor=1.0, map_cap=50)


def test_resolve_map_path_explicit_cap_found(tmp_path):
    paths = _paths(tmp_path)
    snr = paths["quantify_dir"] / "k_1.0_cap_50" / "target_snr.ccp4"
    snr.parent.mkdir(parents=True)
    snr.touch()

    result = _resolve_map_path(paths, k_factor=1.0, map_cap=50)
    assert result == snr


def test_resolve_map_path_explicit_cap_missing_raises(tmp_path):
    paths = _paths(tmp_path)
    paths["quantify_dir"].mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="SNR map not found"):
        _resolve_map_path(paths, k_factor=1.0, map_cap=50)


def test_resolve_map_path_auto_detect_picks_highest_cap(tmp_path):
    paths = _paths(tmp_path)
    for cap in [10, 30, 50]:
        snr = paths["quantify_dir"] / f"k_1.0_cap_{cap}" / "target_snr.ccp4"
        snr.parent.mkdir(parents=True)
        snr.touch()

    result = _resolve_map_path(paths, k_factor=1.0, map_cap=None)
    assert "cap_50" in str(result)


def test_resolve_map_path_auto_detect_no_maps_raises(tmp_path):
    paths = _paths(tmp_path)
    paths["quantify_dir"].mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No SNR map found"):
        _resolve_map_path(paths, k_factor=1.0, map_cap=None)


# ── _resolve_model_path ───────────────────────────────────────────────────────

def test_resolve_model_path_missing_raises(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="Original model not found"):
        _resolve_model_path(paths)


def test_resolve_model_path_exists_returns_path(tmp_path):
    paths = _paths(tmp_path)
    paths["original_pdb"].parent.mkdir(parents=True)
    paths["original_pdb"].touch()

    result = _resolve_model_path(paths)
    assert result == paths["original_pdb"]


# ── _load_null_params ─────────────────────────────────────────────────────────

def test_load_null_params_exact_file_present(tmp_path):
    paths = _paths(tmp_path)
    paths["metadata_dir"].mkdir(parents=True)
    null_file = paths["metadata_dir"] / "target_null_params_k1.0_cap50.json"
    null_file.write_text(json.dumps({"df": 5.0, "loc": 0.0, "scale": 1.0}))

    result = _load_null_params(paths, k_factor=1.0, map_cap=50)
    assert result == {"df": 5.0, "loc": 0.0, "scale": 1.0}


def test_load_null_params_fallback_to_candidate(tmp_path):
    paths = _paths(tmp_path)
    paths["metadata_dir"].mkdir(parents=True)
    # Different k_factor — not the exact match, but a valid fallback.
    fallback = paths["metadata_dir"] / "target_null_params_k2.0_cap50.json"
    fallback.write_text(json.dumps({"df": 3.0, "loc": 0.5, "scale": 0.8}))

    result = _load_null_params(paths, k_factor=1.0, map_cap=50)
    assert result is not None
    assert result["df"] == 3.0


def test_load_null_params_absent_returns_none(tmp_path):
    paths = _paths(tmp_path)
    paths["metadata_dir"].mkdir(parents=True)

    result = _load_null_params(paths, k_factor=1.0, map_cap=50)
    assert result is None


# ── _infer_resolution ─────────────────────────────────────────────────────────

def test_infer_resolution_no_mtz_raises(tmp_path):
    paths = _paths(tmp_path)
    paths["results_dir"].mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No MTZ files found"):
        _infer_resolution(paths)