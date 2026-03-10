import os
from pathlib import Path
from typing import Optional, List, Union, Any

import click
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sqlite3
import pandas as pd

import eliot

from common.logging import setup_eliot_logging
from debias.config import DebiasConfig
from debias.parameter_generator import generate_parameter_files
from common.slurm import (
    generate_preprocessing_sbatch_content,
    generate_omission_sbatch_content,
)

CONF_PATH = Path(__file__).parent.parent / "conf"


def load_debias_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None,
) -> DebiasConfig:
    """
    Function to load, merge, and validate configurations.
    Overrides -> External File -> Internal Defaults.

    Args:
        config_path: Path to an external YAML file.
        overrides: List of dot-notation overrides (e.g. ["debias.run_name=test"]).

    Returns:
        A validated DebiasConfig object.
    """
    GlobalHydra.instance().clear()

    with initialize_config_module(
        version_base=None,
        config_module="conf",
    ):
        cfg = compose(config_name="config", overrides=overrides)

    if config_path:
        print(f"Loading Provided config: {config_path}")
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, user_cfg)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    final_cfg = DebiasConfig(paths=cfg.paths, slurm=cfg.slurm, debias=cfg.debias)
    _config_validator(final_cfg)

    return final_cfg


def run_debias_generation(
    config_path: Optional[Union[str, Path]] = None,
    run_name: Optional[str] = None,
    structure_path: Optional[Union[str, Path]] = None,
    reflections_path: Optional[Union[str, Path]] = None,
    screening_path: Optional[Union[str, Path]] = None,
    work_dir: Optional[Union[str, Path]] = None,
    omit_type: Optional[str] = None,
    omit_fraction: Optional[float] = None,
    always_omit: Optional[str] = None,
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
    slurm_partition: Optional[str] = None,
    slurm_cpus_per_task: Optional[int] = None,
    slurm_mem_per_cpu: Optional[str] = None,
    slurm_num_nodes: Optional[int] = None,
):
    """
    API to generate debias SLURM jobs with explicit parameters.
    """
    overrides = []

    if run_name:
        overrides.append(f"debias.run_name={run_name}")
        overrides.append(f"slurm.job_name={run_name}")
    if structure_path:
        overrides.append(f"debias.structure_path={structure_path}")
    if reflections_path:
        overrides.append(f"debias.reflections_path={reflections_path}")
    if screening_path:
        overrides.append(f"debias.screening_path={screening_path}")
    if work_dir:
        overrides.append(f"paths.work_dir={work_dir}")

    if omit_type:
        overrides.append(f"debias.omit_type={omit_type}")
    if omit_fraction is not None:
        overrides.append(f"debias.omit_fraction={omit_fraction}")
    if iterations is not None:
        overrides.append(f"debias.iterations={iterations}")
    if always_omit:
        overrides.append(f"debias.always_omit={always_omit}")
    if seed is not None:
        overrides.append(f"debias.seed={seed}")

    if slurm_partition:
        overrides.append(f"slurm.partition={slurm_partition}")

    if slurm_cpus_per_task:
        overrides.append(f"slurm.cpus_per_task={slurm_cpus_per_task}")
    if slurm_mem_per_cpu:
        overrides.append(f"slurm.mem_per_cpu={slurm_mem_per_cpu}")
    if slurm_num_nodes:
        overrides.append(f"slurm.num_nodes={slurm_num_nodes}")

    cfg = load_debias_config(config_path=config_path, overrides=overrides)
    generate_slurm_job(cfg)


def _discover_crystals(cfg: DebiasConfig) -> list:
    """Discover (ID, pdb, mtz) tuples from structure_path or screening_path."""
    if cfg.debias.structure_path and cfg.debias.reflections_path:
        return [(
            Path(cfg.debias.structure_path).stem,
            cfg.debias.structure_path,
            cfg.debias.reflections_path,
        )]
    return list(_screening_exploration(
        cfg.debias.screening_path,
            cfg.debias.sqlite_outcomes,
        cfg.debias.max_structures,
    ))


def _setup_debias_directories(
    cfg: DebiasConfig,
    crystals: list,
) -> tuple[dict[str, Path], list[Any]]:
    """Create a directory structure for the run."""

    base = Path(cfg.paths.work_dir) / cfg.debias.run_name

    dirs = {
        "root": base,
        "sbatch": base / "sbatch",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    all_omit_params = []
    for crystal in crystals:
        crystal_name = crystal[0]
        nested = {
            "processed": base / crystal_name / "processed",
            "params": base / crystal_name / "params",
            "results": base / crystal_name / "results",
            "metadata": base / crystal_name / "metadata",
        }
        for nd in nested.values():
            nd.mkdir(parents=True, exist_ok=True)

        params_c = generate_parameter_files(cfg, nested, crystal)
        all_omit_params.extend(params_c)

    return dirs, all_omit_params


def generate_slurm_job(cfg: DebiasConfig):
    """
    Generate SLURM scripts and manifests for cluster submission.
    """
    log_dir = Path(cfg.paths.work_dir) / cfg.debias.run_name / "logs" / "eliot"
    setup_eliot_logging(log_dir, cfg.debias.run_name)

    with eliot.start_action(
        action_type="debias:generate_slurm_job",
        run_name=cfg.debias.run_name,
        work_dir=str(cfg.paths.work_dir),
        omit_type=cfg.debias.omit_type,
        omit_fraction=cfg.debias.omit_fraction,
        iterations=cfg.debias.iterations,
    ):
        crystals = _discover_crystals(cfg)
        click.echo(f"Found {len(crystals)} structure(s) to process.")
        eliot.log_message(
            message_type="debias:structures_found",
            n_structures=len(crystals),
        )

        dirs, omit_params = _setup_debias_directories(cfg, crystals)

        chunk_size = cfg.debias.screening_chunk_size
        n_chunks = max(1, (len(omit_params) + chunk_size - 1) // chunk_size)

        eliot.log_message(
            message_type="debias:setup_complete",
            n_crystals=len(crystals),
            n_omit_params=len(omit_params),
            screening_chunk_size=chunk_size,
            n_omission_chunks=n_chunks,
            sbatch_dir=str(dirs["sbatch"]),
        )

        # Preprocessing: single manifest and script (one job per crystal, always manageable)
        pre_manifest = dirs["sbatch"] / "preprocessing_manifest.txt"
        with open(pre_manifest, "w") as f:
            for crystal in crystals:
                f.write(f"{crystal[0]}|{crystal[1]}|{crystal[2]}\n")

        # Full omit manifest kept for reference
        omit_manifest_full = dirs["sbatch"] / "omit_manifest.txt"
        with open(omit_manifest_full, "w") as f:
            for param_file in omit_params:
                f.write(f"{param_file}\n")

        content_preprocessing = generate_preprocessing_sbatch_content(
            cfg=cfg,
            manifest_path=pre_manifest,
            num_tasks=len(crystals),
            dirs=dirs,
        )
        out_script_preprocessing = dirs["sbatch"] / "submit_preprocessing.slurm"
        with open(out_script_preprocessing, "w") as f:
            f.write(content_preprocessing)
        out_script_preprocessing.chmod(0o755)

        # Omission: split into chunks, one sbatch script per chunk
        omission_scripts = []
        chunks = [omit_params[i:i + chunk_size] for i in range(0, len(omit_params), chunk_size)]
        for i, chunk in enumerate(chunks):
            suffix = f"_{i}" if n_chunks > 1 else ""
            chunk_manifest = dirs["sbatch"] / f"omit_manifest{suffix}.txt"
            with open(chunk_manifest, "w") as f:
                for param_file in chunk:
                    f.write(f"{param_file}\n")

            content_omission = generate_omission_sbatch_content(
                cfg=cfg,
                manifest_path=chunk_manifest,
                num_tasks=len(chunk),
                dirs=dirs,
            )
            out_script_omission = dirs["sbatch"] / f"submit_omission{suffix}.slurm"
            with open(out_script_omission, "w") as f:
                f.write(content_omission)
            out_script_omission.chmod(0o755)
            omission_scripts.append(out_script_omission)

        eliot.log_message(
            message_type="debias:scripts_written",
            preprocessing_script=str(out_script_preprocessing),
            omission_scripts=[str(s) for s in omission_scripts],
        )

        # Build chained submission command
        submission_lines = [f"jid=$(sbatch --parsable {out_script_preprocessing})"]
        for script in omission_scripts:
            submission_lines.append(
                f"jid=$(sbatch --parsable --dependency=afterok:$jid {script})"
            )
        submission_cmd = "\n".join(submission_lines)

        eliot.log_message(
            message_type="debias:submission_command",
            submission_command=submission_cmd,
        )

    click.echo(f"SLURM submission files generated at: {dirs['sbatch']}")
    click.echo(f"Run:\n{submission_cmd}")


def _screening_exploration(
    screening_path: str,
    outcomes: Optional[str] = None,
    max_structures: Optional[int] = None,
):
    """Extract PDB/MTZ pairs from a CSV or Diamond SoakDB SQLite file.

    CSV input always processes all rows. SQLite input supports optional
    outcome filtering and structure count capping.

    outcomes: comma-separated string matched against RefinementOutcome, e.g.
        "CompChem ready, Deposition ready, Deposited"
    Accepted values:
        "Analysis Pending", "PANDDA model - minor", "In Refinement",
        "CompChem ready", "Deposition ready", "Deposited", "Analysed & Rejected"
    """
    # TODO Implement screening exploration logic based on XCA output file structure
    screening_items = []

    if screening_path.endswith(".csv"):
        df = pd.read_csv(screening_path)
        struct_col = next(
            (c for c in ["PDB", "CIF", "structure"] if c in df.columns), None
        )
        if struct_col is None:
            raise ValueError(
                "CSV must contain a 'PDB', 'CIF', or 'structure' column"
            )
        df["ID"] = df[struct_col].map(lambda x: Path(x).stem)
        screening_items = list(zip(df["ID"], df[struct_col], df["MTZ"]))

    if screening_path.endswith(".sqlite"):
        df = _sqlite_as_dataframe(screening_path)

        if outcomes:
            outcome_list = [o.strip() for o in outcomes.split(",")]
            mask = df["RefinementOutcome"].str.contains("|".join(outcome_list), na=False)
            df = df[mask]
            print(f"Outcome filter {outcome_list}: {len(df)} structure(s) retained.")

        ref_pairs = df[
            ["CrystalName", "RefinementPDB_latest", "RefinementMTZ_latest"]
        ].dropna()
        ref_pairs.columns = ["ID", "PDB", "MTZ"]

        dimple_pairs = df[
            ["CrystalName", "DimplePathToPDB", "DimplePathToMTZ"]
        ].dropna()
        dimple_pairs.columns = ["ID", "PDB", "MTZ"]

        final_pairs = ref_pairs.combine_first(dimple_pairs)

        if len(final_pairs) < len(df):
            print(
                f"Skipped {len(df) - len(final_pairs)} datasets: No complete PDB/MTZ pair found."
            )

        if max_structures is not None and len(final_pairs) > max_structures:
            print(
                f"Capping SQLite results from {len(final_pairs)} to {max_structures} structures."
            )
            final_pairs = final_pairs.iloc[:max_structures]

        screening_items.extend(
            zip(final_pairs["ID"], final_pairs["PDB"], final_pairs["MTZ"])
        )

    return screening_items


def _sqlite_as_dataframe(file_path, table="mainTable"):
    conn = sqlite3.connect(file_path)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def _config_validator(cfg: DebiasConfig) -> None:
    """
    Validates the DebiasConfig.
    """
    debias = cfg.debias

    s_path = str(debias.structure_path) if debias.structure_path else None
    r_path = str(debias.reflections_path) if debias.reflections_path else None
    scr_path = str(debias.screening_path) if debias.screening_path else None

    has_screening = scr_path is not None
    has_manual = (s_path is not None) and (r_path is not None)

    if not (has_screening or has_manual):
        raise ValueError(
            "Invalid Configuration: Missing input data sources. \n"
            "You must provide either:\n"
            "  1. 'debias.screening_path' (for batch processing)\n"
            "  2. BOTH 'debias.structure_path' and 'debias.reflections_path' (for single entry processing)"
        )

    if has_manual:
        if not os.path.exists(s_path):
            raise FileNotFoundError(f"Structure file not found at: {s_path}")
        if not os.path.exists(r_path):
            raise FileNotFoundError(f"Reflections file not found at: {r_path}")

    if has_screening:
        if not os.path.exists(scr_path):
            raise FileNotFoundError(f"Screening path not found at: {scr_path}")
