"""
Delta density maps for STOMP realisation ensembles.

A delta map subtracts the density predicted by the *perturbation model* — the
refined model ``{stem}_updated`` that every Phenix omit-map job was run against —
from the STOMP mu map:

    delta(r) = mu(r) - rho_model(r)

Positive delta marks density the model does not explain (unmodelled features,
ordered solvent); negative delta marks model atoms the ensemble does not
support. Two flavours are produced, mirroring the two mu conventions:

    sigma scale:  delta_sigma = mu_sigma - (a * rho_model + b)
    END   scale:  delta_END   = mu_END   - rho_model

The sigma-scaled mu map ({stem}_mean.ccp4) is on an arbitrary, zero-mean scale,
so the model density is least-squares rescaled (a, b) to it over the protein
region before subtraction; the result is a within-dataset residual. The END mu
map ({stem}_end_mean.ccp4) is on the absolute electron-number-density scale
(e-/A^3), so delta_END is a true ensemble-averaged Fo-Fc-style difference map and
no rescaling is applied.

rho_model is computed from {stem}_updated with X-ray form factors
(gemmi.DensityCalculatorX), truncated to the map resolution and placed on the
exact ensemble grid. Its absolute scale carries the atomic F000 term
(end.compute_f000, atomic-only — no bulk solvent), so any bulk-solvent
contribution present in mu_END surfaces as positive delta_END signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import eliot
import gemmi
import numpy as np

from quantify import end as end_module
from quantify.utils import get_experiment_paths, find_experiments, read_mtz

# Grid spacing (A) matching the ensemble maps, which are built by
# utils.read_mtz with transform_f_phi_to_map(sample_rate=3).
_SAMPLE_RATE = 3.0


def compute_model_density(
    structure: gemmi.Structure,
    ref_grid: gemmi.FloatGrid,
    d_min: float,
    sample_rate: float = _SAMPLE_RATE,
) -> np.ndarray:
    """
    Resolution-limited model electron density on the exact ensemble grid

    Builds the calculated electron density of *structure* with X-ray form
    factors, transforms to structure factors, truncates at *d_min* and inverts
    back onto a grid of the exact same shape as *ref_grid*. ``prepare_asu_data``
    drops the F000 term, so the returned density is zero-mean (matching the
    sigma-scaled mu map); the absolute scale is restored separately via
    end.compute_f000 where needed.

    Args:
        structure: The perturbation model ``{stem}_updated``.
        ref_grid: Reference grid (carries cell, space group and shape).
        d_min: High-resolution limit (A) of the map ensemble.
        sample_rate: Oversampling rate; matches utils.read_mtz.

    Returns:
        A float32 array of shape ``ref_grid.shape`` on the absolute e-/A^3 scale
        but with the F000 (mean) term removed.
    """
    dc = gemmi.DensityCalculatorX()
    dc.d_min = d_min
    dc.rate = sample_rate
    dc.set_grid_cell_and_spacegroup(structure)
    dc.put_model_density_on_grid(structure[0])

    recip = gemmi.transform_map_to_f_phi(dc.grid)
    asu = recip.prepare_asu_data(dmin=d_min)
    model_grid = asu.transform_f_phi_to_map(exact_size=list(ref_grid.shape))
    return np.array(model_grid, copy=False).astype(np.float32)


def protein_mask(structure: gemmi.Structure, ref_grid: gemmi.FloatGrid) -> np.ndarray:
    """
    Boolean protein-region mask aligned to the ensemble grid

    Uses a gemmi solvent mask (CCTBX atomic radii) built directly on a grid that
    matches *ref_grid*'s shape, cell and space group, so the returned boolean
    array indexes the same voxels as the mu/model arrays. Waters are removed
    before masking. gemmi marks solvent voxels with value 1, so protein voxels
    are those equal to 0.
    """
    st = structure.clone()
    st.remove_waters()

    nx, ny, nz = ref_grid.shape
    mask = gemmi.FloatGrid(nx, ny, nz)
    mask.set_unit_cell(ref_grid.unit_cell)
    mask.spacegroup = ref_grid.spacegroup

    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.Cctbx)
    masker.put_mask_on_float_grid(mask, st[0])
    return np.array(mask, copy=False) == 0


def fit_linear_scale(
    model: np.ndarray, mu: np.ndarray, mask: np.ndarray
) -> Tuple[float, float]:
    """
    Least-squares fit of model density onto the sigma-scaled mu map

    Solves ``min_{a,b} || a * model + b - mu ||^2`` over the masked voxels. The
    intercept *b* absorbs the zero-mean/F000 offset between the two scales.

    Returns:
        ``(a, b)`` — the slope and intercept of the rescaled model density.
    """
    m = model[mask].astype(np.float64).ravel()
    y = mu[mask].astype(np.float64).ravel()
    if m.size == 0:
        return 1.0, 0.0
    design = np.vstack([m, np.ones_like(m)]).T
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(a), float(b)


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


def _read_map_array(path: Path) -> np.ndarray:
    """Read a CCP4 map written by this package back into a numpy array."""
    ccp4 = gemmi.read_ccp4_map(str(path))
    ccp4.setup(float("nan"))
    return np.array(ccp4.grid, copy=False).astype(np.float32)


def compute_and_write_delta(
    stem: str,
    processed_pdb: Path,
    ref_grid: gemmi.FloatGrid,
    out_dir: Path,
    resolution: float,
    mu_sigma: Optional[np.ndarray] = None,
    do_end: bool = True,
    force: bool = False,
) -> Dict[str, float]:
    """
    Compute and write delta density maps for one experiment

    Shared by the inline (``pseudo-quantify --delta``) and standalone
    (``pseudo-delta``) paths. Always writes the sigma-scaled delta; additionally
    writes the END-scaled delta when ``do_end`` is set and the END mu map
    ({stem}_end_mean.ccp4) is present in *out_dir*.

    Args:
        stem: Experiment stem.
        processed_pdb: Path to the refined model ``{stem}_updated.{pdb,cif}``.
        ref_grid: Reference grid (carries cell + space group + shape).
        out_dir: The ``k_*_cap_*`` quantify directory holding the mu maps.
        resolution: High-resolution limit (A) of the map ensemble.
        mu_sigma: The sigma-scaled mu array. If None, read from
            ``{stem}_mean.ccp4`` in *out_dir*.
        do_end: Also produce the END-scaled delta if the END mu map exists.
        force: Overwrite existing delta outputs.

    Returns:
        Dict of summary scalars (fit coefficients, correlations, F000).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_path = out_dir / f"{stem}_delta_sigma.ccp4"
    if not force and sigma_path.exists():
        print(f"Delta results exist for {stem}. Use --force to overwrite.")
        eliot.log_message(message_type="delta:skipped", stem=stem, reason="results_exist")
        return {}

    mean_map_path = out_dir / f"{stem}_mean.ccp4"
    if mu_sigma is None:
        if not mean_map_path.exists():
            raise FileNotFoundError(
                f"sigma-scaled mu map not found: {mean_map_path}\n"
                f"Run 'pseudo-quantify' for this k/cap first."
            )
        mu_sigma = _read_map_array(mean_map_path)

    structure = gemmi.read_structure(str(processed_pdb))
    model_zero_mean = compute_model_density(structure, ref_grid, resolution)

    # --- sigma-scaled delta -------------------------------------------------
    mask = protein_mask(structure, ref_grid)
    a, b = fit_linear_scale(model_zero_mean, mu_sigma, mask)
    delta_sigma = mu_sigma - (a * model_zero_mean + b)
    _save_ccp4(delta_sigma, ref_grid, sigma_path)

    corr_sigma = float(
        np.corrcoef(model_zero_mean[mask].ravel(), mu_sigma[mask].ravel())[0, 1]
    )

    # --- absolute model density (atomic F000 restored) ----------------------
    cell_volume = ref_grid.unit_cell.volume
    f000_model = end_module.compute_f000(structure, include_bulk=False)
    rho_model = model_zero_mean + np.float32(f000_model / cell_volume)
    _save_ccp4(rho_model, ref_grid, out_dir / f"{stem}_model_density.ccp4")

    summary: Dict[str, float] = {
        "scale_a": a,
        "scale_b": b,
        "corr_sigma": corr_sigma,
        "f000_model": float(f000_model),
        "mean_model_density": float(np.mean(rho_model)),
        "n_protein_voxels": int(mask.sum()),
        "resolution": float(resolution),
        "end_delta_written": False,
    }

    # --- END-scaled delta ---------------------------------------------------
    end_mean_path = out_dir / f"{stem}_end_mean.ccp4"
    if do_end:
        if end_mean_path.exists():
            mu_end = _read_map_array(end_mean_path)
            delta_end = mu_end - rho_model
            _save_ccp4(delta_end, ref_grid, out_dir / f"{stem}_delta_end.ccp4")
            summary["end_delta_written"] = True
            summary["mean_delta_end"] = float(np.mean(delta_end))
        else:
            print(
                f"END mu map not found ({end_mean_path.name}); skipping END-scaled "
                f"delta. Re-run 'pseudo-quantify --end' or 'pseudo-end' first."
            )

    summary_path = out_dir / f"{stem}_delta_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    eliot.log_message(message_type="delta:complete", stem=stem, **summary)
    print(
        f"Delta maps for {stem} written to {out_dir} "
        f"(sigma scale a={a:.4g}, b={b:.4g}, corr={corr_sigma:.3f}"
        + (", END delta on)" if summary["end_delta_written"] else ")")
    )
    return summary


def _run_delta_single(
    paths: dict, k_factor: float, map_cap: Optional[int], force: bool
) -> None:
    """Standalone delta recompute for a single experiment defined by a paths dict."""
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
    if not (out_dir / f"{stem}_mean.ccp4").exists():
        print(
            f"Skipping {stem}: no mu map at {out_dir}. "
            f"Run 'pseudo-quantify' for this k/cap first."
        )
        return

    resolution = gemmi.read_mtz_file(str(res_ref)).resolution_high()
    ref_grid = read_mtz(str(res_ref)).grid

    compute_and_write_delta(
        stem=stem,
        processed_pdb=paths["processed_pdb"],
        ref_grid=ref_grid,
        out_dir=out_dir,
        resolution=resolution,
        mu_sigma=None,
        do_end=True,
        force=force,
    )


def run_delta(
    input_path: Path | str,
    stem: Optional[str] = None,
    force: bool = False,
    k_factor: float = 1.0,
    map_cap: Optional[int] = 50,
) -> None:
    """
    Standalone delta recompute over a completed STOMP run

    Reads each experiment's on-disk mu map(s) and the perturbation model and
    writes delta outputs into the matching ``quantify_results/k_*_cap_*``
    directory, without re-running STOMP. The END-scaled delta is produced only
    where the END mu map already exists.
    """
    input_path = Path(input_path)

    if stem:
        paths = get_experiment_paths(input_path, stem)
        paths["stem"] = stem
        _run_delta_single(paths, k_factor, map_cap, force)
        return

    experiments = list(find_experiments(str(input_path)))
    if not experiments:
        raise ValueError(f"No valid experiments found at {input_path}")

    for paths in experiments:
        _run_delta_single(paths, k_factor, map_cap, force)
