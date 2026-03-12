from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import itertools

import eliot
import gemmi

from debias.phenix_param_parser import ParameterFile
from debias.omission_sampler import stochastic_omission_sampler
from debias.config import DebiasConfig
from debias.omission_table import (
    build_omission_matrix,
    omission_sparse_map,
    save_omission_json,
)

PARAM_TEMPLATES = Path(__file__).parent / "phenix_templates"

# Ordered by preference: Phenix-refined amplitudes first, then CCP4 amplitudes,
# then intensities as last resort.
_F_CANDIDATES: List[Tuple[str, str]] = [
    ("F-obs-filtered", "SIGF-obs-filtered"),
    ("F-obs", "SIGF-obs"),
    ("FP", "SIGFP"),
    ("FOBS", "SIGFOBS"),
    ("Fobs", "SIGFobs"),
    ("F", "SIGF"),
    ("FTOT", "SIGTOT"),
    ("IMEAN", "SIGIMEAN"),
    ("I", "SIGI"),
    ("IOBS", "SIGIOBS"),
]

# Ordered by preference: CCP4 aimless output first, then Phenix, then fallbacks.
# Status is last because it is a character column in some MTZ files and is only
# suitable as a free-set indicator when no dedicated flag column exists.
_RFREE_CANDIDATES: List[str] = [
    "FreeR_flag",
    "FREE",
    "FREER",
    "R-free-flags",
    "Status",
]


def _detect_mtz_labels(
    mtz_path: str,
    crystal_id: str,
) -> Tuple[str, str]:
    """Auto-detect observed-data and R-free labels from an MTZ file.

    Returns ``(f_label, rfree_label)`` where *f_label* is a comma-separated
    amplitude+sigma pair suitable for ``input.xray_data.labels`` and
    *rfree_label* is the R-free flag column name.

    Raises ``ValueError`` with actionable guidance if either label cannot be
    resolved, listing the columns that *were* found so the user knows what to
    set in ``debias.mtz_f_labels`` / ``debias.mtz_rfree_label``.
    """
    with eliot.start_action(
        action_type="debias:detect_mtz_labels",
        crystal_id=crystal_id,
        mtz_path=mtz_path,
    ):
        try:
            mtz = gemmi.read_mtz_file(mtz_path)
        except Exception as exc:
            raise ValueError(
                f"[{crystal_id}] Cannot read MTZ file '{mtz_path}': {exc}"
            ) from exc

        all_cols = [{"label": col.label, "type": col.type} for col in mtz.columns]
        eliot.log_message(
            message_type="debias:mtz_columns_found",
            crystal_id=crystal_id,
            columns=all_cols,
        )

        col_names = {col.label for col in mtz.columns}

        f_label: Optional[str] = None
        for f, sigf in _F_CANDIDATES:
            if f in col_names and sigf in col_names:
                f_label = f"{f},{sigf}"
                break

        rfree_label: Optional[str] = None
        for name in _RFREE_CANDIDATES:
            if name in col_names:
                rfree_label = name
                break

        errors: List[str] = []

        if f_label is None:
            data_cols = sorted(
                col.label for col in mtz.columns if col.type in ("F", "J")
            )
            errors.append(
                f"  No recognised amplitude/intensity pair found.\n"
                f"  F/I columns present: {data_cols or ['(none)']}\n"
                f"  Set 'debias.mtz_f_labels' to e.g. \"FP,SIGFP\" to override."
            )

        if rfree_label is None:
            int_cols = sorted(
                col.label for col in mtz.columns if col.type == "I"
            )
            errors.append(
                f"  No recognised R-free flag column found.\n"
                f"  Integer columns present: {int_cols or ['(none)']}\n"
                f"  Set 'debias.mtz_rfree_label' to e.g. \"FreeR_flag\" to override."
            )

        if errors:
            raise ValueError(
                f"[{crystal_id}] MTZ label detection failed for '{mtz_path}':\n"
                + "\n".join(errors)
            )

        eliot.log_message(
            message_type="debias:mtz_labels_detected",
            crystal_id=crystal_id,
            f_label=f_label,
            rfree_label=rfree_label,
        )

        return f_label, rfree_label


def generate_parameter_files(
    cfg: DebiasConfig, dirs: dict, crystal: Tuple[str, str, str]
) -> List[Path]:
    """Generate parameter files to omit perturbations.

    Args:
        cfg: Module configuration.
        dirs: Dictionary of output directories.
        crystal: Tuple of structure ID, structure, and reflections paths.
    Returns:
        List of generated parameter file paths.
    """
    stem = crystal[0]
    ext = Path(crystal[1]).suffix.lower()  # ".pdb" or ".cif"

    pdb_params = ParameterFile()
    pdb_params.load_from_path(str(PARAM_TEMPLATES / "pdbtools_template.params"))
    pdb_params.set("output.prefix", str(dirs["processed"] / stem))
    pdb_params.save(str(dirs["processed"] / "pdbtools.params"))

    param_ready = ParameterFile()
    param_ready.load_from_path(str(PARAM_TEMPLATES / "ready_set_template.params"))
    param_ready = ParameterFile(PARAM_TEMPLATES / "ready_set_template.params")

    pdb_no_waters = dirs["processed"] / f"{stem}_no_waters{ext}"
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

    # Resolve MTZ labels once per crystal: config overrides take priority,
    # then auto-detection.  Both must be resolved before generating params.
    f_label: Optional[str] = cfg.debias.mtz_f_labels or None
    rfree_label: Optional[str] = cfg.debias.mtz_rfree_label or None

    f_source = "config_override" if f_label is not None else "auto_detected"
    rfree_source = "config_override" if rfree_label is not None else "auto_detected"

    if f_label is None or rfree_label is None:
        detected_f, detected_rfree = _detect_mtz_labels(crystal[2], stem)
        if f_label is None:
            f_label = detected_f
        if rfree_label is None:
            rfree_label = detected_rfree

    eliot.log_message(
        message_type="debias:mtz_labels_resolved",
        crystal_id=stem,
        f_label=f_label,
        f_label_source=f_source,
        rfree_label=rfree_label,
        rfree_label_source=rfree_source,
    )
    print(
        f"[{stem}] MTZ labels — data: {f_label!r} ({f_source})"
        f"  r_free: {rfree_label!r} ({rfree_source})"
    )

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
            "input.pdb.file_name", str(dirs["processed"] / f"{stem}_updated{ext}")
        )
        param_file.set("input.xray_data.file_name", crystal[2])
        param_file.set("input.xray_data.labels", f_label)
        param_file.set("input.xray_data.r_free_flags.label", rfree_label)
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

    if omit_type == "amino_acids":
        for sel in selection_data:
            formatted.append(f"(chain {sel[0]} and resid {sel[1]})")
    elif omit_type == "atoms":
        for sel in selection_data:
            formatted.append(f"(chain {sel[0]} and resid {sel[1]} and name {sel[3]})")
    return " or ".join(formatted)
