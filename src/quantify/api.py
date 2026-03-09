from __future__ import annotations

import json
import numpy as np
import gemmi
import eliot
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import time

from common.logging import setup_eliot_logging
from quantify import ownership_logic
from quantify import aggregator

from quantify import statistical_model
from quantify.utils import (
    get_experiment_paths,
    infer_omission_mode,
    find_experiments,
    read_mtz,
)


def load_ensemble(results_dir: Path, stem: str, map_cap: Optional[int]):
    """
    Loads MTZ files into a 4D numpy array.
    """

    first_path = results_dir / f"{stem}_0" / f"{stem}_0.mtz"
    if not first_path.exists():
        raise FileNotFoundError("First map not found")

    ref_map = read_mtz(str(first_path))
    ref_grid = ref_map.grid
    shape = ref_grid.shape

    ensemble_array = np.zeros((map_cap, *shape), dtype=np.float32)

    for i in range(map_cap):
        p = results_dir / f"{stem}_{i}" / f"{stem}_{i}.mtz"
        if p.exists():
            temp_grid = read_mtz(str(p)).grid
            ensemble_array[i] = temp_grid.array

        else:
            print(f"Warning: Missing map {i} at {p}. Skipping.")
            continue

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


def _quantify_single(paths: dict, force: bool, k_factor: float, map_cap: Optional[int]):
    """Run quantification for a single experiment defined by a paths dict."""
    stem = paths["stem"]
    log_dir = paths["root"] / "logs" / "eliot"
    setup_eliot_logging(log_dir, stem)

    with eliot.start_action(
        action_type="quantify:run",
        stem=stem,
        k_factor=k_factor,
        map_cap=map_cap,
    ):
        start_time = time.time()

        current_map_cap = map_cap
        if current_map_cap is None:
            files = list(paths["results_dir"].glob(f"*/{stem}_*.mtz"))
            current_map_cap = len(files)

        out_dir = paths["quantify_dir"] / f"k_{k_factor}_cap_{current_map_cap}"
        out_dir.mkdir(exist_ok=True, parents=True)

        if not force and (out_dir / f"{stem}_snr.ccp4").exists():
            print(f"Results exist for {stem}. Use --force to overwrite.")
            eliot.log_message(message_type="quantify:skipped", stem=stem, reason="results_exist")
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

        eliot.log_message(
            message_type="quantify:params",
            resolution_angstrom=round(resolution, 3),
            omission_mode=mode,
            map_cap=current_map_cap,
        )

        with eliot.start_action(action_type="quantify:build_spatial_index"):
            spatial_index = ownership_logic.build_spatial_index(
                paths["processed_pdb"], omission_map, resolution, k_factor, mode
            )
        if not spatial_index:
            print("Error: Spatial index failed.")
            return

        with eliot.start_action(action_type="quantify:load_ensemble", map_cap=current_map_cap):
            data, grid = load_ensemble(paths["results_dir"], stem, current_map_cap)

        nx, ny, nz = data.shape[1:]
        with eliot.start_action(
            action_type="quantify:aggregate_ensemble",
            grid_shape=[nx, ny, nz],
            n_voxels=nx * ny * nz,
        ):
            sig, nos, snr = aggregator.aggregate_ensemble(data, grid, spatial_index)

        save_map(sig, grid, out_dir / f"{stem}_mean.ccp4")
        save_map(nos, grid, out_dir / f"{stem}_std.ccp4")
        save_map(snr, grid, out_dir / f"{stem}_snr.ccp4")

        print("Running statistical modeling")

        with eliot.start_action(action_type="quantify:statistical_model", n_samples=20000):
            null_samples = statistical_model.sample_null_distribution(
                snr_map=str(out_dir / f"{stem}_snr.ccp4"),
                model_path=paths["original_pdb"],
                n_samples=20000,
            )

            # Persist null distribution parameters in metadata so downstream analysis
            # can derive per-alpha significance thresholds without re-sampling.
            # Filename encodes the quantification run they correspond to.
            null_params = statistical_model.fit_null_distribution(null_samples)
            null_params_path = (
                paths["metadata_dir"]
                / f"{stem}_null_params_k{k_factor}_cap{current_map_cap}.json"
            )
            with open(null_params_path, "w") as fh:
                json.dump(null_params, fh, indent=2)

            p_value_map = statistical_model.fit_t_test(null_samples, snr)
            save_map(np.array(p_value_map), grid, out_dir / f"{stem}_p_value.ccp4")

        elapsed_time = time.time() - start_time
        eliot.log_message(
            message_type="quantify:complete",
            stem=stem,
            output_dir=str(out_dir),
            elapsed_seconds=round(elapsed_time, 2),
        )
        print(f"Quantification results of the {stem} perturbation saved to {out_dir}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")


def run_quantification(
    input_path: Path | str,
    stem: Optional[str] = None,
    force: bool = False,
    k_factor: float = 1.0,
    map_cap: Optional[int] = 50,
    num_processes: int = 1,
):
    input_path = Path(input_path)

    if stem:
        # Explicit stem: treat input_path as a single experiment root.
        paths = get_experiment_paths(input_path, stem)
        paths["stem"] = stem
        _quantify_single(paths, force, k_factor, map_cap)
        return

    experiments = list(find_experiments(str(input_path)))
    if not experiments:
        raise ValueError(f"No valid experiments found at {input_path}")

    if len(experiments) == 1:
        _quantify_single(experiments[0], force, k_factor, map_cap)
    else:
        print(
            f"Screening mode: {len(experiments)} experiments found, "
            f"using {num_processes} process(es)."
        )
        worker = partial(_quantify_single, force=force, k_factor=k_factor, map_cap=map_cap)
        with Pool(max(1, num_processes)) as pool:
            pool.map(worker, experiments)
