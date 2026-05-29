from __future__ import annotations

import json
from pathlib import Path

import gemmi
import numpy as np
import pytest

from quantify import end

DATA = Path(__file__).resolve().parents[1] / "test_data"
MODEL = DATA / "test_model.pdb"

# Independent element -> Z table (mirrors the reference's atomsf.lib lookup),
# used to cross-check compute_f000 against a separate code path.
_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "P": 15}


@pytest.fixture(scope="module")
def structure():
    return gemmi.read_structure(str(MODEL))


def _independent_f000_atomic(structure):
    n_sym = len(structure.find_spacegroup().operations())
    asu = 0.0
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                asu += atom.occ * _Z[atom.element.name]
    return n_sym * asu


# --- compute_f000 -----------------------------------------------------------


def test_f000_atomic_matches_independent_sum(structure):
    expected = _independent_f000_atomic(structure)
    got = end.compute_f000(structure, include_bulk=False)
    assert abs(got - expected) / expected <= 1e-3


def test_f000_bulk_adds_ksol_times_solvent_volume(structure):
    k_sol = 0.35
    f_no = end.compute_f000(structure, include_bulk=False)
    f_bulk = end.compute_f000(structure, {"k_sol": k_sol}, include_bulk=True)
    v_solv = end._solvent_volume(structure)
    assert (f_bulk - f_no) == pytest.approx(k_sol * v_solv, rel=1e-6)
    assert f_bulk > f_no


def test_f000_include_bulk_requires_ksol(structure):
    with pytest.raises(ValueError, match="k_sol"):
        end.compute_f000(structure, bulk_solvent_params=None, include_bulk=True)


# --- run config -------------------------------------------------------------


def test_load_run_config_missing(tmp_path):
    with pytest.raises(end.RunConfigError, match="Run configuration not found"):
        end.load_run_config(tmp_path, "foo")


def test_load_run_config_malformed(tmp_path):
    (tmp_path / "foo_run_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(end.RunConfigError, match="malformed"):
        end.load_run_config(tmp_path, "foo")


def test_load_run_config_missing_required_key(tmp_path):
    (tmp_path / "foo_run_config.json").write_text(
        json.dumps({"x": 1}), encoding="utf-8"
    )
    with pytest.raises(end.RunConfigError, match="use_bulk_and_scaling"):
        end.load_run_config(tmp_path, "foo")


def test_load_run_config_ok(tmp_path):
    (tmp_path / "foo_run_config.json").write_text(
        json.dumps({"use_bulk_and_scaling": True, "bulk_solvent_k_sol": 0.4}),
        encoding="utf-8",
    )
    cfg = end.load_run_config(tmp_path, "foo")
    assert cfg["use_bulk_and_scaling"] is True


# --- resolve_run_config (legacy-run fallback) -------------------------------


def test_resolve_run_config_missing_defaults_no_bulk(tmp_path):
    cfg = end.resolve_run_config(tmp_path, "foo")
    assert cfg["use_bulk_and_scaling"] is False
    assert cfg["bulk_solvent_k_sol"] == end.DEFAULT_K_SOL


def test_resolve_run_config_malformed_still_errors(tmp_path):
    (tmp_path / "foo_run_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(end.RunConfigError, match="malformed"):
        end.resolve_run_config(tmp_path, "foo")


def test_resolve_run_config_present_ok(tmp_path):
    (tmp_path / "foo_run_config.json").write_text(
        json.dumps({"use_bulk_and_scaling": True, "bulk_solvent_k_sol": 0.4}),
        encoding="utf-8",
    )
    assert end.resolve_run_config(tmp_path, "foo")["use_bulk_and_scaling"] is True


# --- omission inversion & per-realisation model -----------------------------


def test_realisation_omit_sets_inverts_and_clips():
    omission_map = {"a": [0, 2], "b": [1], "c": [5]}  # index 5 out of range
    sets = end.realisation_omit_sets(omission_map, n_maps=3)
    assert sets == [{"a"}, {"b"}, {"a"}]


def test_build_realisation_model_removes_only_omitted(structure):
    # take keys for the first two atoms of the structure
    keys = []
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                keys.append(
                    end._atom_key(
                        chain.name,
                        residue.seqid.num,
                        residue.name,
                        atom.name,
                        str(atom.altloc),
                    )
                )
    omit = set(keys[:2])
    n_before = sum(1 for c in structure[0] for r in c for _ in r)
    model_k = end.build_realisation_model(structure, omit)
    n_after = sum(1 for c in model_k[0] for r in c for _ in r)
    assert n_after == n_before - 2
    # original untouched
    assert sum(1 for c in structure[0] for r in c for _ in r) == n_before
    # fewer electrons in M^(k)
    assert end.compute_f000(model_k) < end.compute_f000(structure)


# --- END scaling math -------------------------------------------------------


def test_end_scale_sets_per_map_mean_to_f000_over_v():
    rng = np.random.default_rng(0)
    n, nx, ny, nz = 4, 6, 5, 7
    data = rng.standard_normal((n, nx, ny, nz)).astype(np.float32)
    data -= data.mean(axis=(1, 2, 3), keepdims=True)  # zero-mean per map
    f000 = np.array([1000.0, 1200.0, 900.0, 1100.0])
    cell_volume = 5000.0
    end_data = end.end_scale_ensemble(data, f000, cell_volume)
    per_map_mean = end_data.mean(axis=(1, 2, 3))
    assert np.allclose(per_map_mean, f000 / cell_volume, atol=1e-4)


# --- integration: compute_and_write_end -------------------------------------


def _make_experiment(tmp_path, *, use_bulk, k_sol=0.35, n_maps=3):
    """Build a minimal experiment tree (model + metadata) for END tests."""
    stem = "xtl"
    processed = tmp_path / "processed"
    metadata = tmp_path / "metadata"
    processed.mkdir()
    metadata.mkdir()

    structure = gemmi.read_structure(str(MODEL))
    structure.write_pdb(str(processed / f"{stem}_updated.pdb"))

    # omit the first atom in realisation 0 only
    first_key = None
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                first_key = end._atom_key(
                    chain.name,
                    residue.seqid.num,
                    residue.name,
                    atom.name,
                    str(atom.altloc),
                )
                break
            break
        break
    (metadata / f"{stem}_omission_map.json").write_text(
        json.dumps({first_key: [0]}), encoding="utf-8"
    )
    (metadata / f"{stem}_run_config.json").write_text(
        json.dumps({"use_bulk_and_scaling": use_bulk, "bulk_solvent_k_sol": k_sol}),
        encoding="utf-8",
    )

    grid = gemmi.FloatGrid()
    grid.setup_from(structure, spacing=3.0)
    grid.spacegroup = structure.find_spacegroup()
    shape = grid.shape

    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_maps, *shape)).astype(np.float32)
    data -= data.mean(axis=(1, 2, 3), keepdims=True)
    return stem, processed, metadata, data, grid


def test_compute_and_write_end_mean_equals_f000_over_v(tmp_path):
    stem, processed, metadata, data, grid = _make_experiment(tmp_path, use_bulk=False)
    out_dir = tmp_path / "out"
    summary = end.compute_and_write_end(
        stem=stem,
        processed_pdb=processed / f"{stem}_updated.pdb",
        metadata_dir=metadata,
        omission_json=metadata / f"{stem}_omission_map.json",
        ensemble_data=data,
        ref_grid=grid,
        out_dir=out_dir,
        force=True,
        write_realisations=False,
    )
    assert (out_dir / f"{stem}_end_mean.ccp4").exists()
    assert (out_dir / f"{stem}_end_std.ccp4").exists()
    assert (out_dir / f"{stem}_end_snr.ccp4").exists()

    mean_grid = gemmi.read_ccp4_map(str(out_dir / f"{stem}_end_mean.ccp4")).grid
    spatial_mean = float(np.array(mean_grid, copy=False).mean())
    expected = summary["f000_mean"] / grid.unit_cell.volume
    assert spatial_mean == pytest.approx(expected, abs=1e-3)
    assert summary["include_bulk"] is False


def test_bulk_on_shifts_mean_by_bulk_f000(tmp_path):
    # Same synthetic ensemble, two run configs differing only in bulk solvent.
    stem, processed, metadata, data, grid = _make_experiment(tmp_path, use_bulk=False)
    out_off = tmp_path / "off"
    s_off = end.compute_and_write_end(
        stem,
        processed / f"{stem}_updated.pdb",
        metadata,
        metadata / f"{stem}_omission_map.json",
        data,
        grid,
        out_off,
        force=True,
        write_realisations=False,
    )
    # flip the persisted config to bulk-on and recompute
    (metadata / f"{stem}_run_config.json").write_text(
        json.dumps({"use_bulk_and_scaling": True, "bulk_solvent_k_sol": 0.35}),
        encoding="utf-8",
    )
    out_on = tmp_path / "on"
    s_on = end.compute_and_write_end(
        stem,
        processed / f"{stem}_updated.pdb",
        metadata,
        metadata / f"{stem}_omission_map.json",
        data,
        grid,
        out_on,
        force=True,
        write_realisations=False,
    )
    cell_volume = grid.unit_cell.volume
    delta_mean = s_on["mean_density"] - s_off["mean_density"]
    expected = (s_on["f000_mean"] - s_off["f000_mean"]) / cell_volume
    assert delta_mean == pytest.approx(expected, rel=1e-4)
    assert s_on["f000_mean"] > s_off["f000_mean"]  # bulk adds electrons


def test_standalone_matches_inline_byte_identical(tmp_path):
    # Build a real MTZ ensemble so run_end loads it the same way quantify does.
    stem = "xtl"
    root = tmp_path / stem
    (root / "processed").mkdir(parents=True)
    (root / "metadata").mkdir()
    results = root / "results"

    structure = gemmi.read_structure(str(MODEL))
    structure.write_pdb(str(root / "processed" / f"{stem}_updated.pdb"))
    (root / "metadata" / f"{stem}_omission_map.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (root / "metadata" / f"{stem}_run_config.json").write_text(
        json.dumps({"use_bulk_and_scaling": False, "bulk_solvent_k_sol": 0.35}),
        encoding="utf-8",
    )

    n_maps = 3
    import shutil

    for i in range(n_maps):
        d = results / f"{stem}_{i}"
        d.mkdir(parents=True)
        shutil.copy(DATA / "test_reflections.mtz", d / f"{stem}_{i}.mtz")

    from quantify.api import load_ensemble
    from quantify.utils import get_experiment_paths

    data, grid = load_ensemble(results, stem, n_maps)

    inline_dir = tmp_path / "inline"
    end.compute_and_write_end(
        stem,
        root / "processed" / f"{stem}_updated.pdb",
        root / "metadata",
        root / "metadata" / f"{stem}_omission_map.json",
        data,
        grid,
        inline_dir,
        force=True,
    )

    end.run_end(root, stem=stem, force=True, k_factor=1.0, map_cap=n_maps)
    standalone_dir = (
        get_experiment_paths(root, stem)["quantify_dir"] / f"k_1.0_cap_{n_maps}"
    )

    for name in (
        f"{stem}_end_mean.ccp4",
        f"{stem}_end_std.ccp4",
        f"{stem}_end_snr.ccp4",
    ):
        assert (inline_dir / name).read_bytes() == (standalone_dir / name).read_bytes()


def test_standalone_missing_run_config_defaults_no_bulk(tmp_path):
    stem = "xtl"
    root = tmp_path / stem
    (root / "processed").mkdir(parents=True)
    (root / "metadata").mkdir()
    results = root / "results"
    structure = gemmi.read_structure(str(MODEL))
    structure.write_pdb(str(root / "processed" / f"{stem}_updated.pdb"))
    (root / "metadata" / f"{stem}_omission_map.json").write_text("{}", encoding="utf-8")

    import shutil

    d = results / f"{stem}_0"
    d.mkdir(parents=True)
    shutil.copy(DATA / "test_reflections.mtz", d / f"{stem}_0.mtz")

    # NOTE: deliberately no {stem}_run_config.json — a legacy run. END must not
    # fail; it falls back to no bulk-solvent modelling and writes its maps.
    from quantify.utils import get_experiment_paths

    end.run_end(root, stem=stem, force=True, k_factor=1.0, map_cap=1)

    out_dir = get_experiment_paths(root, stem)["quantify_dir"] / "k_1.0_cap_1"
    for name in (
        f"{stem}_end_mean.ccp4",
        f"{stem}_end_std.ccp4",
        f"{stem}_end_snr.ccp4",
    ):
        assert (out_dir / name).exists()
