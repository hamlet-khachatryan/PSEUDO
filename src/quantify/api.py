from __future__ import annotations

import numpy as np
import gemmi
from pathlib import Path
from typing import Optional

import time

from quantify import ownership_logic
from quantify import aggregator

from quantify import statistical_model
from quantify.utils import (
    get_experiment_paths,
    infer_stem,
    infer_omission_mode,
    read_mtz,
)


def load_ensemble(results_dir: Path, stem: str, map_cap: Optional[int]):
    """
    Loads MTZ files into a 4D numpy array.
    """
    ensemble = []
    for i in range(map_cap):
        p = results_dir / f"{stem}_{i}" / f"{stem}_{i}.mtz"

        if not p.exists():
            print(f"Warning: Missing map {i} at {p}. Skipping.")
            continue

        ensemble.append(read_mtz(str(p)).grid)

    ensemble_array = np.stack([i.array for i in ensemble])
    ref_grid = ensemble[0]

    return ensemble_array, ref_grid


def save_map(array: np.ndarray, ref_grid: gemmi.FloatGrid, output_path: Path | str):
    """
    Saves MTZ files as a ccp4 map.
    """
    grid = gemmi.FloatGrid(array)
    grid.set_unit_cell(ref_grid.unit_cell)
    grid.spacegroup = ref_grid.spacegroup

    ccp4_map = gemmi.Ccp4Map()
    ccp4_map.grid = grid
    ccp4_map.update_ccp4_header()
    ccp4_map.write_ccp4_map(str(output_path))


def run_quantification(
    input_path: Path | str,
    stem: Optional[str],
    force: bool,
    k_factor: float,
    map_cap: Optional[int],
):
    start_time = time.time()

    if not stem:
        stem = infer_stem(Path(input_path) / "processed")
        if not stem:
            raise ValueError("Could not infer stem. Specify --stem.")

    paths = get_experiment_paths(input_path, stem)

    if map_cap is None:
        files = list(paths["results_dir"].glob(f"*/{stem}_*.mtz"))
        map_cap = len(files)

    out_dir = paths["quantify_dir"] / f"k_{k_factor}_cap_{map_cap}"
    out_dir.mkdir(exist_ok=True, parents=True)

    if not force and (out_dir / f"{stem}_snr.ccp4").exists():
        print(f"Results exist for {stem}. Use --force to overwrite.")
        return

    print(f"--- Quantifying {stem} (K={k_factor}) ---")

    if not paths["omission_json"].exists():
        raise FileNotFoundError(f"JSON not found: {paths['omission_json']}")

    omission_map = ownership_logic.load_omission_map(paths["omission_json"])
    mode = infer_omission_mode(omission_map)

    res_ref = paths["results_dir"] / f"{stem}_0" / f"{stem}_0.mtz"

    if not res_ref.exists():
        raise FileNotFoundError("Cannot find map 0 to infer resolution.")

    mtz = gemmi.read_mtz_file(str(res_ref))
    resolution = mtz.resolution_high()
    print(f"Resolution: {resolution:.2f} Å")

    spatial_index = ownership_logic.build_spatial_index(
        paths["processed_pdb"], omission_map, resolution, k_factor, mode
    )
    if not spatial_index:
        print("Error: Spatial index failed.")
        return

    data, grid = load_ensemble(paths["results_dir"], stem, map_cap)
    sig, nos, snr, dists = aggregator.aggregate_ensemble(data, grid, spatial_index)

    save_map(sig, grid, out_dir / f"{stem}_mean.ccp4")
    save_map(nos, grid, out_dir / f"{stem}_std.ccp4")
    save_map(snr, grid, out_dir / f"{stem}_snr.ccp4")
    np.save(out_dir / f"{stem}_distributions.npy", dists, allow_pickle=True)

    print("Running statistical modeling")

    null_samples = statistical_model.sample_null_distribution(
        snr_map=str(out_dir / f"{stem}_snr.ccp4"),
        model_path=paths["original_pdb"],
        n_samples=20000,
    )
    p_value_map = statistical_model.fit_t_test(null_samples, snr)
    save_map(np.array(p_value_map), grid, out_dir / f"{stem}_p_value.ccp4")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Quantification results of the {stem} perturbation saved to {out_dir}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
