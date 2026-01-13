from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import itertools

from debias.phenix_param_parser import ParameterFile
from debias.omission_sampler import stochastic_omission_sampler
from debias.config import DebiasConfig
from debias.omission_table import (
    build_omission_matrix,
    omission_sparse_map,
    save_omission_json,
)

PARAM_TEMPLATES = Path(__file__).parent / "phenix_templates"


def generate_parameter_files(
    cfg: DebiasConfig, dirs: dict, crystal: Tuple[str, str, str]
) -> List[Path]:
    """Generate parameter files for omit perturbations.

    Args:
        cfg: Module configuration.
        dirs: Dictionary of output directories.
        crystal: Tuple of structure ID, structure and reflections paths.
    Returns:
        List of generated parameter file paths.
    """
    stem = crystal[0]

    pdb_params = ParameterFile()
    pdb_params.load_from_path(str(PARAM_TEMPLATES / "pdbtools_template.params"))
    pdb_params.set("output.prefix", str(dirs["processed"] / stem))
    pdb_params.save(str(dirs["processed"] / "pdbtools.params"))

    param_ready = ParameterFile()
    param_ready.load_from_path(str(PARAM_TEMPLATES / "ready_set_template.params"))
    param_ready = ParameterFile(PARAM_TEMPLATES / "ready_set_template.params")

    pdb_no_waters = dirs["processed"] / f"{stem}_no_waters.pdb"
    output_updated = dirs["processed"] / f"{stem}_updated"

    param_ready.set("ready_set.input.pdb_file_name", str(pdb_no_waters))
    param_ready.set("ready_set.input.output_dir", str(dirs["processed"]) + "/")
    param_ready.set("ready_set.input.cif_dir_name", str(dirs["processed"]) + "/")
    param_ready.set(
        "ready_set.input.cif_file_name", str(dirs["processed"] / "cif_file.cif")
    )
    param_ready.set("ready_set.input.output_file_name", str(output_updated))
    param_ready.set(
        "ready_set.input.ligand_cache_directory", str(dirs["processed"]) + "/"
    )
    param_ready.save(str(dirs["processed"] / "ready_set.params"))

    generated_files = []
    selections = stochastic_omission_sampler(
        structure_path=crystal[1],
        omit_type=cfg.debias.omit_type,
        omit_fraction=cfg.debias.omit_fraction,
        n_iterations=cfg.debias.iterations,
        always_omit=cfg.debias.always_omit,
        seed=cfg.debias.seed,
    )
    flattened = set(list(itertools.chain.from_iterable(selections)))
    sorted_ids = sorted(list(flattened), key=lambda x: x[1])

    id_strings, mat = build_omission_matrix(sorted_ids, selections)
    sparse_map = omission_sparse_map(id_strings, mat)
    save_omission_json(dirs["metadata"] / f"{stem}_omission_map.json", sparse_map)

    for i, selection in enumerate(selections):
        run_id = f"{stem}_{i}"

        job_result_dir = dirs["results"] / run_id
        job_result_dir.mkdir(parents=True, exist_ok=True)

        param_file = ParameterFile()
        param_file.load_from_path(str(PARAM_TEMPLATES / "maps_template.params"))

        param_file.set(
            "input.pdb.file_name", str(dirs["processed"] / f"{stem}_updated.pdb")
        )
        param_file.set("input.xray_data.file_name", crystal[2])
        param_file.set("output.file_name", str(job_result_dir / f"{run_id}.mtz"))
        param_file.set("output.job_title", run_id)

        formatted_sel = _format_selection(selection, cfg.debias.omit_type)
        param_file.set("omit_map.boxing.selection", formatted_sel)

        out_path = dirs["params"] / f"{run_id}.params"
        param_file.save(str(out_path))
        generated_files.append(out_path)

    return generated_files


def _format_selection(selection_data: List[Tuple], omit_type: str) -> str:
    """Helper to format the selection list into a Phenix string."""
    formatted = []

    if omit_type == "residues":
        for sel in selection_data:
            formatted.append(f"(chain {sel[0]} and resid {sel[1]})")
    elif omit_type == "atoms":
        for sel in selection_data:
            formatted.append(f"(chain {sel[0]} and resid {sel[1]} and name {sel[3]})")
    return " or ".join(formatted)
