from __future__ import annotations

import shutil
from pathlib import Path

import gemmi
import numpy as np
import pytest

from quantify import delta
from quantify.utils import read_mtz

DATA = Path(__file__).resolve().parents[1] / "test_data"
MODEL = DATA / "test_model.pdb"
MTZ = DATA / "test_reflections.mtz"


@pytest.fixture(scope="module")
def ref_grid():
    return read_mtz(str(MTZ)).grid


@pytest.fixture(scope="module")
def resolution():
    return gemmi.read_mtz_file(str(MTZ)).resolution_high()


@pytest.fixture(scope="module")
def structure():
    return gemmi.read_structure(str(MODEL))


# --- compute_model_density --------------------------------------------------


def test_model_density_matches_ensemble_grid(structure, ref_grid, resolution):
    dens = delta.compute_model_density(structure, ref_grid, resolution)
    assert dens.shape == ref_grid.shape
    assert dens.dtype == np.float32


def test_model_density_is_zero_mean(structure, ref_grid, resolution):
    # prepare_asu_data drops F000, so the returned density has ~zero mean.
    dens = delta.compute_model_density(structure, ref_grid, resolution)
    assert abs(float(dens.mean())) < 1e-3


def test_model_density_correlates_with_experimental_map(
    structure, ref_grid, resolution
):
    dens = delta.compute_model_density(structure, ref_grid, resolution)
    exp = np.nan_to_num(np.array(ref_grid, copy=False))
    corr = np.corrcoef(dens.ravel(), exp.ravel())[0, 1]
    assert corr > 0.5  # model density tracks the 2FoFc map


# --- protein_mask -----------------------------------------------------------


def test_protein_mask_shape_and_nontrivial(structure, ref_grid):
    mask = delta.protein_mask(structure, ref_grid)
    assert mask.shape == ref_grid.shape
    assert mask.dtype == bool
    # some protein, but not the whole cell
    assert 0 < mask.sum() < mask.size


# --- fit_linear_scale -------------------------------------------------------


def test_fit_linear_scale_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    model = rng.standard_normal((8, 8, 8)).astype(np.float32)
    mask = np.ones_like(model, dtype=bool)
    a_true, b_true = 2.5, -0.7
    mu = a_true * model + b_true
    a, b = delta.fit_linear_scale(model, mu, mask)
    assert a == pytest.approx(a_true, rel=1e-4)
    assert b == pytest.approx(b_true, abs=1e-4)


def test_fit_linear_scale_empty_mask_returns_identity():
    model = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.zeros_like(model, dtype=bool)
    a, b = delta.fit_linear_scale(model, model, mask)
    assert (a, b) == (1.0, 0.0)


# --- integration: compute_and_write_delta -----------------------------------


def _seed_quantify_dir(tmp_path, *, with_end: bool):
    """Build a minimal quantify output dir with a mu map (and optionally END mu)."""
    stem = "xtl"
    processed = tmp_path / "processed"
    processed.mkdir()
    structure = gemmi.read_structure(str(MODEL))
    structure.write_pdb(str(processed / f"{stem}_updated.pdb"))

    ref_grid = read_mtz(str(MTZ)).grid
    resolution = gemmi.read_mtz_file(str(MTZ)).resolution_high()

    out_dir = tmp_path / "k_1.0_cap_3"
    out_dir.mkdir()

    # a sigma-scaled mu map: just use the experimental 2FoFc map as a stand-in
    mu_sigma = np.nan_to_num(np.array(ref_grid, copy=False)).astype(np.float32)
    delta._save_ccp4(mu_sigma, ref_grid, out_dir / f"{stem}_mean.ccp4")

    if with_end:
        # absolute-scale mu: model density + a small offset
        model = delta.compute_model_density(structure, ref_grid, resolution)
        f000 = 1.0  # arbitrary positive offset / V proxy
        mu_end = model + np.float32(f000)
        delta._save_ccp4(mu_end, ref_grid, out_dir / f"{stem}_end_mean.ccp4")

    return stem, processed, out_dir, ref_grid, resolution


def test_compute_and_write_delta_sigma_only(tmp_path):
    stem, processed, out_dir, ref_grid, resolution = _seed_quantify_dir(
        tmp_path, with_end=False
    )
    summary = delta.compute_and_write_delta(
        stem=stem,
        processed_pdb=processed / f"{stem}_updated.pdb",
        ref_grid=ref_grid,
        out_dir=out_dir,
        resolution=resolution,
        mu_sigma=None,
        do_end=True,
        force=True,
    )
    assert (out_dir / f"{stem}_delta_sigma.ccp4").exists()
    assert (out_dir / f"{stem}_model_density.ccp4").exists()
    assert (out_dir / f"{stem}_delta_summary.json").exists()
    # END mu absent -> no END delta
    assert not (out_dir / f"{stem}_delta_end.ccp4").exists()
    assert summary["end_delta_written"] is False


def test_compute_and_write_delta_with_end(tmp_path):
    stem, processed, out_dir, ref_grid, resolution = _seed_quantify_dir(
        tmp_path, with_end=True
    )
    summary = delta.compute_and_write_delta(
        stem=stem,
        processed_pdb=processed / f"{stem}_updated.pdb",
        ref_grid=ref_grid,
        out_dir=out_dir,
        resolution=resolution,
        mu_sigma=None,
        do_end=True,
        force=True,
    )
    assert (out_dir / f"{stem}_delta_end.ccp4").exists()
    assert summary["end_delta_written"] is True


def test_delta_missing_mu_raises(tmp_path):
    stem = "xtl"
    processed = tmp_path / "processed"
    processed.mkdir()
    gemmi.read_structure(str(MODEL)).write_pdb(str(processed / f"{stem}_updated.pdb"))
    out_dir = tmp_path / "k_1.0_cap_3"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="mu map not found"):
        delta.compute_and_write_delta(
            stem=stem,
            processed_pdb=processed / f"{stem}_updated.pdb",
            ref_grid=read_mtz(str(MTZ)).grid,
            out_dir=out_dir,
            resolution=2.0,
            mu_sigma=None,
            do_end=True,
            force=True,
        )


def test_delta_skips_when_results_exist(tmp_path, capsys):
    stem, processed, out_dir, ref_grid, resolution = _seed_quantify_dir(
        tmp_path, with_end=False
    )
    (out_dir / f"{stem}_delta_sigma.ccp4").write_bytes(b"")  # pre-existing
    summary = delta.compute_and_write_delta(
        stem=stem,
        processed_pdb=processed / f"{stem}_updated.pdb",
        ref_grid=ref_grid,
        out_dir=out_dir,
        resolution=resolution,
        force=False,
    )
    assert summary == {}
    assert "Delta results exist" in capsys.readouterr().out


# --- standalone == inline ---------------------------------------------------


def test_standalone_matches_inline(tmp_path):
    stem = "xtl"
    root = tmp_path / stem
    (root / "processed").mkdir(parents=True)
    (root / "metadata").mkdir()
    results = root / "results"

    structure = gemmi.read_structure(str(MODEL))
    structure.write_pdb(str(root / "processed" / f"{stem}_updated.pdb"))
    (root / "metadata" / f"{stem}_omission_map.json").write_text("{}", encoding="utf-8")

    n_maps = 3
    for i in range(n_maps):
        d = results / f"{stem}_{i}"
        d.mkdir(parents=True)
        shutil.copy(MTZ, d / f"{stem}_{i}.mtz")

    # Seed the mu map the way quantify would, into the k_*_cap_* dir.
    out_dir = root / "quantify_results" / f"k_1.0_cap_{n_maps}"
    out_dir.mkdir(parents=True)
    ref_grid = read_mtz(str(MTZ)).grid
    resolution = gemmi.read_mtz_file(str(MTZ)).resolution_high()
    mu_sigma = np.nan_to_num(np.array(ref_grid, copy=False)).astype(np.float32)
    delta._save_ccp4(mu_sigma, ref_grid, out_dir / f"{stem}_mean.ccp4")

    # inline-style direct call
    inline_dir = tmp_path / "inline"
    inline_dir.mkdir()
    delta._save_ccp4(mu_sigma, ref_grid, inline_dir / f"{stem}_mean.ccp4")
    delta.compute_and_write_delta(
        stem=stem,
        processed_pdb=root / "processed" / f"{stem}_updated.pdb",
        ref_grid=ref_grid,
        out_dir=inline_dir,
        resolution=resolution,
        mu_sigma=mu_sigma,
        do_end=False,
        force=True,
    )

    # standalone
    delta.run_delta(root, stem=stem, force=True, k_factor=1.0, map_cap=n_maps)

    a = (inline_dir / f"{stem}_delta_sigma.ccp4").read_bytes()
    b = (out_dir / f"{stem}_delta_sigma.ccp4").read_bytes()
    assert a == b
