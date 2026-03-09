from __future__ import annotations

import json
import time
import eliot
from dataclasses import replace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import gemmi

from common.logging import setup_eliot_logging
from analyse.muse.config import AggregationConfig
from analyse.muse.pipeline import (
    export_atom_csv,
    export_residue_csv,
    export_summary,
    run_muse,
    write_scored_pdb,
)
from analyse.muse.config import snr_map_config
from quantify.statistical_model import compute_significance_threshold
from quantify.utils import find_experiments, get_experiment_paths


_DEFAULT_K_FACTOR = 1.0
_DEFAULT_MAP_CAP = 50


def _resolve_map_path(paths: dict, k_factor: float, map_cap: Optional[int]) -> Path:
    """Locate the SNR CCP4 map for the given k_factor/map_cap. Auto-detects highest cap if map_cap is None."""
    stem = paths["stem"]
    qdir = paths["quantify_dir"]

    if not qdir.exists():
        raise FileNotFoundError(
            f"No quantify_results directory found at {qdir}. "
            "Run 'pseudo quantify' before 'pseudo analyse'."
        )

    if map_cap is not None:
        snr_path = qdir / f"k_{k_factor}_cap_{map_cap}" / f"{stem}_snr.ccp4"
        if not snr_path.exists():
            raise FileNotFoundError(
                f"SNR map not found at {snr_path}. "
                f"Verify that quantification ran with k={k_factor} and cap={map_cap}, "
                "or omit --map_cap to auto-detect."
            )
        return snr_path

    candidates = sorted(qdir.glob(f"k_{k_factor}_cap_*"))
    for candidate in reversed(candidates):  # prefer highest cap
        snr_path = candidate / f"{stem}_snr.ccp4"
        if snr_path.exists():
            return snr_path

    default_snr = qdir / f"k_{_DEFAULT_K_FACTOR}_cap_{_DEFAULT_MAP_CAP}" / f"{stem}_snr.ccp4"
    if default_snr.exists():
        return default_snr

    raise FileNotFoundError(
        f"No SNR map found in {qdir} for k_factor={k_factor}. "
        "Run 'pseudo-debias' followed by 'pseudo-quantify' to generate maps, "
        "or provide a map explicitly with --map_path."
    )


def _resolve_model_path(paths: dict) -> Path:
    """Return the processed model path, raising FileNotFoundError if absent."""
    model_path = paths["processed_pdb"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Processed model not found at {model_path}. "
            "Run 'pseudo-debias' first to generate the processed structure."
        )
    return model_path


def _load_null_params(paths: dict, k_factor: float, map_cap: int) -> Optional[dict]:
    """Load fitted null-distribution parameters from metadata. Returns None if not found."""
    null_params_path = (
        paths["metadata_dir"]
        / f"{paths['stem']}_null_params_k{k_factor}_cap{map_cap}.json"
    )
    if not null_params_path.exists():
        candidates = list(paths["metadata_dir"].glob(f"{paths['stem']}_null_params_k*.json"))
        if candidates:
            null_params_path = candidates[0]
            print(
                f"Warning: null params for k={k_factor} cap={map_cap} not found. "
                f"Using {null_params_path.name} instead."
            )
        else:
            print(
                "Warning: No null distribution parameters found in metadata/. "
                "Significance threshold will use MUSE paper defaults. "
                "Re-run 'pseudo quantify' to generate null params."
            )
            return None
    with open(null_params_path) as fh:
        return json.load(fh)


def _infer_resolution(paths: dict) -> float:
    """Infer resolution in Ångströms from the first available MTZ file."""
    stem = paths["stem"]
    results_dir = paths["results_dir"]

    canonical = results_dir / f"{stem}_0" / f"{stem}_0.mtz"
    if canonical.exists():
        return gemmi.read_mtz_file(str(canonical)).resolution_high()

    all_mtz = list(results_dir.glob("*/*.mtz"))
    if not all_mtz:
        raise FileNotFoundError(
            f"No MTZ files found in {results_dir}. "
            "Run 'pseudo-debias' to generate omission map ensembles."
        )
    return gemmi.read_mtz_file(str(all_mtz[0])).resolution_high()

def _analyse_single(
    paths: dict,
    map_path: Optional[str],
    model_path: Optional[str],
    k_factor: float,
    map_cap: Optional[int],
    significance_alpha: float = 0.05,
) -> None:
    """Run MUSE analysis for one experiment and write results to analyse_results/."""
    stem = paths["stem"]
    log_dir = paths["root"] / "logs" / "eliot"
    setup_eliot_logging(log_dir, stem)

    with eliot.start_action(
        action_type="analyse:run",
        stem=stem,
        k_factor=k_factor,
        map_cap=map_cap,
        significance_alpha=significance_alpha,
    ):
        start_time = time.time()

        if map_path:
            resolved_map = Path(map_path)
            if not resolved_map.exists():
                raise FileNotFoundError(f"Provided map not found: {map_path}")
        else:
            resolved_map = _resolve_map_path(paths, k_factor, map_cap)

        if model_path:
            resolved_model = Path(model_path)
            if not resolved_model.exists():
                raise FileNotFoundError(f"Provided model not found: {model_path}")
        else:
            resolved_model = _resolve_model_path(paths)

        resolution = _infer_resolution(paths)

        out_dir = paths["root"] / "analyse_results"
        out_dir.mkdir(exist_ok=True, parents=True)

        print(
            f"--- Analysing {stem} | map: {resolved_map.name} "
            f"| model: {resolved_model.name} | resolution: {resolution:.2f} Å ---"
        )

        eliot.log_message(
            message_type="analyse:params",
            map=resolved_map.name,
            model=resolved_model.name,
            resolution_angstrom=round(resolution, 3),
        )

        null_params = _load_null_params(paths, k_factor, map_cap if map_cap is not None else _DEFAULT_MAP_CAP)
        significance_threshold = None
        base_config = snr_map_config()
        if null_params is not None:
            significance_threshold = compute_significance_threshold(null_params, alpha=significance_alpha)
            config = replace(
                base_config,
                aggregation=AggregationConfig(
                    opia_threshold=significance_threshold,
                    missing_density_threshold=significance_threshold,
                ),
            )
        else:
            config = base_config

        eliot.log_message(
            message_type="analyse:significance_threshold",
            threshold=significance_threshold,
            source="null_distribution" if null_params is not None else "paper_defaults",
        )

        with eliot.start_action(action_type="analyse:run_muse"):
            result = run_muse(str(resolved_map), str(resolved_model), resolution, config=config)

        export_atom_csv(result, str(out_dir / f"{stem}_atoms.csv"))
        export_residue_csv(result, str(out_dir / f"{stem}_residues.csv"))

        summary = export_summary(result)
        summary["significance_alpha"] = significance_alpha
        summary["significance_snr_threshold"] = significance_threshold
        with open(out_dir / f"{stem}_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)

        write_scored_pdb(result, str(resolved_model), str(out_dir / f"{stem}_scored.pdb"))

        elapsed = time.time() - start_time
        eliot.log_message(
            message_type="analyse:complete",
            opia=round(result.opia, 4),
            n_atoms=len(result.atom_scores),
            n_residues=len(result.residue_scores),
            elapsed_seconds=round(elapsed, 2),
            output_dir=str(out_dir),
        )
        print(
            f"Analysis results saved to {out_dir}\n"
            f"  OPIA: {result.opia:.3f} | atoms: {len(result.atom_scores)} "
            f"| residues: {len(result.residue_scores)} | time: {elapsed:.1f}s"
        )

def run_analysis(
    input_path: str,
    stem: Optional[str] = None,
    map_path: Optional[str] = None,
    model_path: Optional[str] = None,
    k_factor: float = _DEFAULT_K_FACTOR,
    map_cap: Optional[int] = _DEFAULT_MAP_CAP,
    num_processes: int = 1,
    significance_alpha: float = 0.05,
) -> None:
    """Run MUSE analysis on a single experiment or a screening directory.
    Auto-detects single vs. screening mode; parallelises with num_processes in screening.
    """
    input_path = Path(input_path)

    if stem:
        # Explicit stem: single experiment at input_path.
        paths = get_experiment_paths(input_path, stem)
        paths["stem"] = stem
        _analyse_single(paths, map_path, model_path, k_factor, map_cap, significance_alpha)
        return

    experiments = list(find_experiments(str(input_path)))
    if not experiments:
        raise ValueError(
            f"No valid experiments found at {input_path}. "
            "Ensure the directory contains a 'processed/' subdirectory with a "
            "'*_updated.pdb' or '*_updated.cif' file and a corresponding "
            "'metadata/*_omission_map.json'."
        )

    if len(experiments) == 1:
        _analyse_single(experiments[0], map_path, model_path, k_factor, map_cap, significance_alpha)
    else:
        print(
            f"Screening mode: {len(experiments)} experiments found, "
            f"using {num_processes} process(es)."
        )
        worker = partial(
            _analyse_single,
            map_path=map_path,
            model_path=model_path,
            k_factor=k_factor,
            map_cap=map_cap,
            significance_alpha=significance_alpha,
        )
        with Pool(max(1, num_processes)) as pool:
            pool.map(worker, experiments)
