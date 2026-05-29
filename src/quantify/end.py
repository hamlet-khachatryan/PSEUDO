"""
END (Electron Number Density) maps for STOMP realisation ensembles.

STOMP realisation maps are computed with the h=0 (F000) Fourier term dropped,
so each realisation map rho^(k)(r) has zero spatial mean. END mode restores the
absolute electron-number-density scale by injecting the per-realisation total
electron count F000^(k):

    rho_END^(k)(r) = F000^(k) / V + rho^(k)(r)

where V is the unit-cell volume. F000^(k) is computed *per realisation* from the
model M^(k) (the refined model with realisation k's omit set removed):

    use_bulk_and_scaling = False:  F000^(k) = sum_j q_j Z_j  (over M^(k), x n_sym)
    use_bulk_and_scaling = True:   F000^(k) = atomic sum + k_sol * V_solvent^(k)

The bulk-solvent decision is read from the persisted run configuration written
by the debias stage. If that file is absent (legacy runs predating run-config
support, all produced without bulk solvent) END defaults to no bulk solvent; a
config that is present but malformed is still an error. The resulting mu_S_END,
sigma_S_END and SNR_END maps live on the absolute e-/A^3 scale and are distinct
statistics from the sigma-scaled MUSE maps (which are for within-dataset
thresholding).

Reference:
    Lang PT, Holton JM, Fraser JS, Alber T (2014).
    Protein structural ensembles are revealed by redefining X-ray electron
    density noise. PNAS 111(1):237-242.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import eliot
import gemmi
import numpy as np

from quantify import aggregator
from quantify.utils import get_experiment_paths, find_experiments

# Default flat bulk-solvent density (e-/A^3) used if the run config omits it.
DEFAULT_K_SOL = 0.35

# Grid spacing (A) for the analytic bulk-solvent mask volume calculation.
_MASK_SPACING = 0.6


class RunConfigError(RuntimeError):
    """Raised when a run's persisted configuration is missing or malformed."""


def load_run_config(metadata_dir: Path | str, stem: str) -> Dict:
    """
    Load the persisted debias run configuration for a crystal

    The configuration records how the STOMP realisations were produced, most
    importantly whether bulk-solvent modelling and scaling were enabled. END
    computation relies on this to decide whether the bulk-solvent term is
    included in F000; it is never inferred.

    Args:
        metadata_dir: The crystal's ``metadata/`` directory.
        stem: The experiment stem.

    Returns:
        The parsed configuration dict.

    Raises:
        RunConfigError: If the file is absent, unreadable, or missing the
            required ``use_bulk_and_scaling`` key (e.g. runs predating END
            support).
    """
    path = Path(metadata_dir) / f"{stem}_run_config.json"
    if not path.exists():
        raise RunConfigError(
            f"Run configuration not found: {path}\n"
            f"END computation needs the bulk-solvent convention used during "
            f"refinement. Re-run 'pseudo-debias generate-params' to produce it, "
            f"or this run predates END support and cannot be processed safely."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise RunConfigError(f"Run configuration is malformed: {path}: {exc}") from exc

    if "use_bulk_and_scaling" not in data:
        raise RunConfigError(
            f"Run configuration {path} is missing the required "
            f"'use_bulk_and_scaling' key."
        )
    return data


def resolve_run_config(metadata_dir: Path | str, stem: str) -> Dict:
    """
    Load the run config, defaulting to no-bulk-solvent when it is ABSENT

    Runs predating run-config support have no ``metadata/{stem}_run_config.json``.
    Every such run was produced without bulk-solvent modelling, so END falls back
    to that convention rather than failing. A config file that is PRESENT but
    malformed or missing the required key is still a hard error (via
    :func:`load_run_config`) — that signals corruption, not a legacy run.

    Args:
        metadata_dir: The crystal's ``metadata/`` directory.
        stem: The experiment stem.

    Returns:
        The parsed configuration dict, or a no-bulk default when the file is
        absent.
    """
    path = Path(metadata_dir) / f"{stem}_run_config.json"
    if not path.exists():
        eliot.log_message(
            message_type="end:run_config_missing_default",
            stem=stem,
            assumed_use_bulk_and_scaling=False,
        )
        print(
            f"No run configuration for {stem}; assuming no bulk-solvent "
            f"modelling (legacy STOMP run)."
        )
        return {"use_bulk_and_scaling": False, "bulk_solvent_k_sol": DEFAULT_K_SOL}
    return load_run_config(metadata_dir, stem)


def _atom_key(
    chain_name: str, seqid_num: int, resname: str, atom_name: str, altloc: str
) -> str:
    """
    Reconstruct an omission-map atom key for a model atom

    Matches the key construction used by the debias stage
    (``omission_sampler.extract_ids`` -> ``omission_table._id_to_str``) so that
    keys built here can be looked up directly against the persisted omission
    map.
    """
    return "|".join((chain_name, str(seqid_num), resname, atom_name, altloc))


def realisation_omit_sets(
    omission_map: Dict[str, List[int]], n_maps: int
) -> List[Set[str]]:
    """
    Invert the omission map into per-realisation omitted-atom-key sets

    Args:
        omission_map: ``{atom_key: [realisation indices where omitted]}``.
        n_maps: Number of realisations to build sets for (indices 0..n_maps-1).

    Returns:
        A list of length ``n_maps``; element k is the set of atom keys omitted
        in realisation k.
    """
    omit_sets: List[Set[str]] = [set() for _ in range(n_maps)]
    for atom_key, indices in omission_map.items():
        for k in indices:
            if 0 <= k < n_maps:
                omit_sets[k].add(atom_key)
    return omit_sets


def build_realisation_model(
    updated_structure: gemmi.Structure, omit_keys: Set[str]
) -> gemmi.Structure:
    """
    Build the realisation model M^(k) by removing the omitted atoms

    Returns an independent clone of *updated_structure* with every atom whose
    omission-map key is in *omit_keys* removed. The input is not modified.
    """
    model = updated_structure.clone()
    for gmodel in model:
        for chain in gmodel:
            for residue in chain:
                doomed = [
                    i
                    for i, atom in enumerate(residue)
                    if _atom_key(
                        chain.name,
                        residue.seqid.num,
                        residue.name,
                        atom.name,
                        str(atom.altloc),
                    )
                    in omit_keys
                ]
                for i in reversed(doomed):
                    del residue[i]
    return model


def _solvent_volume(
    structure: gemmi.Structure, spacing: float = _MASK_SPACING
) -> float:
    """
    Analytic bulk-solvent volume (A^3) of the unit cell for a model

    Uses a gemmi solvent mask (CCTBX atomic radii). The mask grid is built from
    the structure, so the grid's space group is applied and symmetry-related
    copies of the model are masked out; the solvent region is the complement.

    Note: in gemmi the solvent mask marks solvent voxels with value 1.
    """
    grid = gemmi.FloatGrid()
    grid.setup_from(structure, spacing=spacing)
    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    masker.put_mask_on_float_grid(grid, structure[0])
    arr = np.array(grid, copy=False)
    solvent_fraction = float((arr == 1).mean())
    return solvent_fraction * structure.cell.volume


def compute_f000(
    refined_model: gemmi.Structure,
    bulk_solvent_params: Optional[Dict[str, float]] = None,
    include_bulk: bool = False,
) -> float:
    """
    Total electrons F000 in the unit cell for a refined model

    Computes the atomic-electron sum over the asymmetric unit (occupancy-weighted
    atomic numbers) scaled by the number of symmetry operators to give the
    whole-cell electron count. When *include_bulk* is set, a flat bulk-solvent
    term ``k_sol * V_solvent`` is added, consistent with refinements that used
    bulk-solvent modelling and scaling.

    Args:
        refined_model: The (per-realisation) model M^(k) as a gemmi.Structure.
        bulk_solvent_params: Dict with key ``k_sol`` (e-/A^3). Required when
            *include_bulk* is True.
        include_bulk: Whether to add the bulk-solvent contribution.

    Returns:
        F000 in electrons.

    Reference:
        Lang, Holton, Fraser, Alber (2014) PNAS 111:237-242.
    """
    spacegroup = refined_model.find_spacegroup()
    n_sym = len(spacegroup.operations()) if spacegroup is not None else 1

    asu_electrons = 0.0
    for chain in refined_model[0]:
        for residue in chain:
            for atom in residue:
                asu_electrons += atom.occ * atom.element.atomic_number

    f000 = n_sym * asu_electrons

    if include_bulk:
        if not bulk_solvent_params or "k_sol" not in bulk_solvent_params:
            raise ValueError(
                "include_bulk=True requires bulk_solvent_params with a 'k_sol' key."
            )
        k_sol = float(bulk_solvent_params["k_sol"])
        f000 += k_sol * _solvent_volume(refined_model)

    return f000


def per_realisation_f000(
    updated_structure: gemmi.Structure,
    omit_sets: List[Set[str]],
    include_bulk: bool,
    k_sol: float,
) -> np.ndarray:
    """
    Compute F000^(k) for every realisation

    F000 is recomputed per realisation from M^(k); it is never cached or reused
    across realisations.
    """
    bulk_params = {"k_sol": k_sol} if include_bulk else None
    values = np.empty(len(omit_sets), dtype=np.float64)
    for k, omit_keys in enumerate(omit_sets):
        model_k = build_realisation_model(updated_structure, omit_keys)
        values[k] = compute_f000(
            model_k, bulk_solvent_params=bulk_params, include_bulk=include_bulk
        )
    return values


def end_scale_ensemble(
    ensemble_data: np.ndarray, f000_per_realisation: np.ndarray, cell_volume: float
) -> np.ndarray:
    """
    Add F000^(k)/V to each realisation map to form the END-scale ensemble

    Args:
        ensemble_data: Array (n_maps, nx, ny, nz) of zero-mean realisation maps.
        f000_per_realisation: Array (n_maps,) of F000 values (electrons).
        cell_volume: Unit-cell volume (A^3).

    Returns:
        A new array (n_maps, nx, ny, nz) on the absolute e-/A^3 scale.
    """
    offsets = (f000_per_realisation / cell_volume).astype(np.float32)
    return ensemble_data + offsets[:, None, None, None]


def _save_ccp4(
    array: np.ndarray, ref_grid: gemmi.FloatGrid, output_path: Path | str
) -> None:
    """Write a numpy array as a CCP4 map using a reference grid's cell/symmetry."""
    grid = gemmi.FloatGrid(np.ascontiguousarray(array, dtype=np.float32))
    grid.set_unit_cell(ref_grid.unit_cell)
    grid.spacegroup = ref_grid.spacegroup
    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = grid
    ccp4_map.update_ccp4_header()
    ccp4_map.write_ccp4_map(str(output_path))


def compute_and_write_end(
    stem: str,
    processed_pdb: Path,
    metadata_dir: Path,
    omission_json: Path,
    ensemble_data: np.ndarray,
    ref_grid: gemmi.FloatGrid,
    out_dir: Path,
    force: bool = False,
    write_realisations: bool = True,
) -> Dict[str, float]:
    """
    Compute and write END maps for one experiment

    Shared by the inline (``pseudo-quantify --end``) and standalone
    (``pseudo-end``) paths. Reads the persisted run configuration to decide the
    bulk-solvent convention, computes F000^(k) per realisation, forms the
    END-scale ensemble, accumulates mu/sigma/SNR on the absolute e-/A^3 scale,
    and writes the maps alongside the existing sigma-scaled outputs.

    Args:
        stem: Experiment stem.
        processed_pdb: Path to the refined model ``{stem}_updated.{pdb,cif}``.
        metadata_dir: The crystal's ``metadata/`` directory.
        omission_json: Path to ``{stem}_omission_map.json``.
        ensemble_data: Array (n_maps, nx, ny, nz) of zero-mean realisation maps.
        ref_grid: Reference grid (carries cell + space group).
        out_dir: Destination directory (the ``k_*_cap_*`` quantify dir).
        force: Overwrite existing END outputs.
        write_realisations: Also write each per-realisation rho_END^(k) map.

    Returns:
        Dict with summary scalars ('f000_mean', 'mean_density', 'include_bulk').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    snr_path = out_dir / f"{stem}_end_snr.ccp4"
    if not force and snr_path.exists():
        print(f"END results exist for {stem}. Use --force to overwrite.")
        eliot.log_message(message_type="end:skipped", stem=stem, reason="results_exist")
        return {}

    run_cfg = resolve_run_config(metadata_dir, stem)
    include_bulk = bool(run_cfg["use_bulk_and_scaling"])
    k_sol = float(run_cfg.get("bulk_solvent_k_sol", DEFAULT_K_SOL))

    with omission_json.open("r", encoding="utf-8") as f:
        omission_map = json.load(f)

    n_maps = ensemble_data.shape[0]
    omit_sets = realisation_omit_sets(omission_map, n_maps)

    structure = gemmi.read_structure(str(processed_pdb))
    f000 = per_realisation_f000(structure, omit_sets, include_bulk, k_sol)

    cell_volume = ref_grid.unit_cell.volume
    end_data = end_scale_ensemble(ensemble_data, f000, cell_volume)

    mean_map, std_map, snr_map = aggregator.aggregate_ensemble(end_data, ref_grid, None)

    _save_ccp4(mean_map, ref_grid, out_dir / f"{stem}_end_mean.ccp4")
    _save_ccp4(std_map, ref_grid, out_dir / f"{stem}_end_std.ccp4")
    _save_ccp4(snr_map, ref_grid, snr_path)

    if write_realisations:
        for k in range(n_maps):
            _save_ccp4(end_data[k], ref_grid, out_dir / f"{stem}_end_rho_{k}.ccp4")

    summary = {
        "f000_mean": float(np.mean(f000)),
        "f000_min": float(np.min(f000)),
        "f000_max": float(np.max(f000)),
        "mean_density": float(np.mean(mean_map)),
        "include_bulk": include_bulk,
        "k_sol": k_sol if include_bulk else None,
        "n_maps": int(n_maps),
    }
    eliot.log_message(message_type="end:complete", stem=stem, **summary)
    print(
        f"END maps for {stem} written to {out_dir} "
        f"(bulk_solvent={'on' if include_bulk else 'off'}, "
        f"mean rho = {summary['mean_density']:.4f} e-/A^3)"
    )
    return summary


def _run_end_single(
    paths: dict, k_factor: float, map_cap: Optional[int], force: bool
) -> None:
    """Standalone END recompute for a single experiment defined by a paths dict."""
    from quantify.api import load_ensemble  # lazy import to avoid an import cycle

    stem = paths["stem"]

    current_map_cap = map_cap
    if current_map_cap is None:
        files = list(paths["results_dir"].glob(f"*/{stem}_*.mtz"))
        current_map_cap = len(files)

    res_ref = paths["results_dir"] / f"{stem}_0" / f"{stem}_0.mtz"
    if not res_ref.exists():
        print(f"Skipping {stem}: no perturbation maps found.")
        return

    out_dir = paths["quantify_dir"] / f"k_{k_factor}_cap_{current_map_cap}"

    data, grid = load_ensemble(paths["results_dir"], stem, current_map_cap)
    compute_and_write_end(
        stem=stem,
        processed_pdb=paths["processed_pdb"],
        metadata_dir=paths["metadata_dir"],
        omission_json=paths["omission_json"],
        ensemble_data=data,
        ref_grid=grid,
        out_dir=out_dir,
        force=force,
    )


def run_end(
    input_path: Path | str,
    stem: Optional[str] = None,
    force: bool = False,
    k_factor: float = 1.0,
    map_cap: Optional[int] = 50,
) -> None:
    """
    Standalone END recompute over a completed STOMP run

    Reads each experiment's persisted run configuration and realisation maps and
    writes END outputs into the matching ``quantify_results/k_*_cap_*`` directory,
    without re-running STOMP. Errors clearly if the run configuration is missing.
    """
    input_path = Path(input_path)

    if stem:
        paths = get_experiment_paths(input_path, stem)
        paths["stem"] = stem
        _run_end_single(paths, k_factor, map_cap, force)
        return

    experiments = list(find_experiments(str(input_path)))
    if not experiments:
        raise ValueError(f"No valid experiments found at {input_path}")

    for paths in experiments:
        _run_end_single(paths, k_factor, map_cap, force)
