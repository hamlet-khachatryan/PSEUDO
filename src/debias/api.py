import os
from pathlib import Path
from typing import Optional, List, Union, Any

import click
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import sqlite3
import pandas as pd

from src.debias.config import DebiasConfig
from src.debias.parameter_generator import generate_parameter_files
from src.common.slurm import (
    generate_preprocessing_sbatch_content,
    generate_omission_sbatch_content,
)

CONF_PATH = Path(__file__).parent.parent / "conf"
QUERY_SQLITE_DIAMOND = "(RefinementOutcome == '4 - CompChem ready') or (RefinementOutcome == '5 - Deposition ready') or (RefinementOutcome == '6 - Deposited')"


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

    with initialize(
        version_base=None,
        config_path=str(os.path.relpath(CONF_PATH, Path(__file__).parent)),
    ):
        base_cfg = compose(config_name="config")

    cfg = base_cfg

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
    work_dir: Optional[Union[str, Path]] = None,
    omit_type: Optional[str] = None,
    omit_fraction: Optional[float] = None,
    iterations: Optional[int] = None,
    seed: Optional[int] = None,
    slurm_partition: Optional[str] = None,
    slurm_cpus_per_task: Optional[int] = None,
    slurm_mem_per_cpu: Optional[str] = None,
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
    if work_dir:
        overrides.append(f"paths.work_dir={work_dir}")

    if omit_type:
        overrides.append(f"debias.omit_type={omit_type}")
    if omit_fraction is not None:
        overrides.append(f"debias.omit_fraction={omit_fraction}")
    if iterations is not None:
        overrides.append(f"debias.iterations={iterations}")
    if seed is not None:
        overrides.append(f"debias.seed={seed}")

    if slurm_partition:
        overrides.append(f"slurm.partition={slurm_partition}")

    if slurm_cpus_per_task:
        overrides.append(f"slurm.cpus_per_task={slurm_cpus_per_task}")
    if slurm_mem_per_cpu:
        overrides.append(f"slurm.mem_per_cpu={slurm_mem_per_cpu}")

    cfg = load_debias_config(config_path=config_path, overrides=overrides)
    generate_slurm_job(cfg)


def _setup_debias_directories(
    cfg: DebiasConfig,
) -> tuple[dict[str, Path], list[Any], list[Any]]:
    """Create directory structure for the run."""

    crystals = []

    if cfg.debias.structure_path and cfg.debias.reflections_path:
        crystals.append(
            (
                Path(cfg.debias.structure_path).stem,
                cfg.debias.structure_path,
                cfg.debias.reflections_path,
            )
        )
    elif cfg.debias.screening_path:
        crystals.extend(_screening_exploration(cfg.debias.screening_path))

    base = Path(cfg.paths.work_dir) / cfg.debias.run_name

    dirs = {
        "root": base,
        "sbatch": base / "sbatch",
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

    return dirs, crystals, all_omit_params


def generate_slurm_job(cfg: DebiasConfig):
    """
    Generate SLURM scripts and manifests for cluster submission.
    """
    dirs, crystals, omit_params = _setup_debias_directories(cfg)

    pre_manifest = dirs["sbatch"] / "preprocessing_manifest.txt"
    omit_manifest = dirs["sbatch"] / "omit_manifest.txt"

    with open(pre_manifest, "w") as f:
        for crystal in crystals:
            f.write(f"{crystal[0]}|{crystal[1]}|{crystal[2]}\n")

    with open(omit_manifest, "w") as f:
        for param_file in omit_params:
            f.write(f"{param_file}\n")

    content_preprocessing = generate_preprocessing_sbatch_content(
        cfg=cfg,
        manifest_path=pre_manifest,
        num_tasks=len(crystals),
        dirs=dirs,
    )

    content_omission = generate_omission_sbatch_content(
        cfg=cfg,
        manifest_path=omit_manifest,
        num_tasks=len(omit_params),
    )

    out_script_preprocessing = dirs["sbatch"] / "submit_preprocessing.slurm"
    out_script_omission = dirs["sbatch"] / "submit_omission.slurm"

    with open(out_script_preprocessing, "w") as f:
        f.write(content_preprocessing)

    with open(out_script_omission, "w") as f:
        f.write(content_omission)

    out_script_preprocessing.chmod(0o755)
    out_script_omission.chmod(0o755)

    click.echo(f"SLURM submission files generated at: {dirs['sbatch']}")
    click.echo(
        f"Run : jid=$(sbatch --parsable {out_script_preprocessing}) && sbatch --dependency=afterok:$jid {out_script_omission}"
    )


def _screening_exploration(screening_path: str):
    """Analyze the input screening results file and extract relevant PDB and MTZ file pairs.

    The extraction depends on the file format (CSV or SQLite database).
    The function supports CSV files containing two specific columns ('PDB', 'MTZ')
    and SQLite databases with a predefined query structure corresponding to Diamond
    soakdb files.

    Args:
        screening_path (str): Path to the screening file (CSV or SQLite).

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the path to a PDB file
            and the path to the corresponding MTZ file.
    """
    # TODO Implement screening exploration logic based on XCA output file structure
    screening_items = []

    if screening_path.endswith(".csv"):
        df = pd.read_csv(screening_path)
        df["ID"] = df["PDB"].map(lambda x: Path(x).stem)
        screening_items = list(zip(df["ID"], df["PDB"], df["MTZ"]))

    if screening_path.endswith(".sqlite"):
        df = _sqlite_as_dataframe(screening_path).query(QUERY_SQLITE_DIAMOND)

        ref_pairs = df[
            ["CrystalName", "RefinementPDB_latest", "RefinementMTZ_latest"]
        ].dropna()
        ref_pairs.columns = ["ID", "PDB", "MTZ"]

        dimple_pairs = df[
            ["CrystalName", "DimplePathToPDB", "DimplePathToMTZ"]
        ].dropna()
        dimple_pairs.columns = ["ID", "PDB", "MTZ"]

        final_pairs = ref_pairs.combine_first(dimple_pairs)

        screening_items.extend(
            zip(final_pairs["ID"], final_pairs["PDB"], final_pairs["MTZ"])
        )

        if len(final_pairs) < len(df):
            print(
                f"Skipped {len(df) - len(final_pairs)} datasets: No complete PDB/MTZ pair found."
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
