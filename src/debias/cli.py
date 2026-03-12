import click
from debias.api import generate_slurm_job, load_debias_config


@click.group(name="debias")
def debias_cli():
    """Commands for the Debias/Omit Map module."""
    pass


@debias_cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to external YAML configuration file.",
)
@click.option("--run_name", type=str, help="Run name.")
@click.option("--structure_path", type=click.Path(exists=True), help="Input structure path (.pdb or .cif).")
@click.option(
    "--reflections_path", type=click.Path(exists=True), help="Input MTZ path."
)
@click.option(
    "--screening_path",
    type=click.Path(exists=True),
    help="Path to the screening file. CSV with 'PDB' and 'MTZ' column names. SoakDB sql file for the DLS XChem Data.",
)
@click.option("--work_dir", type=click.Path(exists=True), help="Working directory.")
# Omission Parameters

@click.option(
    "--omit_type",
    type=click.Choice(["amino_acids", "atoms"]),
    help="Type of model elements to omit.",
)
@click.option(
    "--omit_fraction",
    type=float,
    help="Fraction of structure to omit per iteration.",
)
@click.option(
    "--always_omit",
    type=str,
    help="String of comma seperated selections to always omit. E.g. 'A 567, A 234', 'B 123 CA' ",
)
@click.option("--iterations", type=int, help="Number of omission iterations.")
@click.option("--seed", type=int, help="Random seed for reproducibility.")
# SQLite screening options
@click.option(
    "--sqlite_outcomes",
    type=str,
    default=None,
    help=(
        "Comma-separated substrings to match against RefinementOutcome in SoakDB SQLite files. "
        "Accepted values: 'CompChem ready', 'Deposition ready', 'Deposited', 'Analysed & Rejected'."
        "No effect on CSV input. Defaults to no filtering (all structures included)."
    ),
)
@click.option(
    "--max_structures",
    type=int,
    default=None,
    help=(
        "Maximum number of structures to process from a SQLite file. "
        "No effect on CSV or single-structure input. "
        "Defaults to no cap (all matching structures processed)."
    ),
)
@click.option(
    "--screening_chunk_size",
    type=int,
    default=None,
    help=(
        "Maximum number of omission jobs per sbatch array submission. "
        "Omission jobs are chunked and submitted sequentially to avoid flooding the scheduler. "
        "Defaults to 1000."
    ),
)
# MTZ label overrides
@click.option(
    "--mtz_f_labels",
    type=str,
    default=None,
    help=(
        "Comma-separated amplitude+sigma column labels to use as observed data, "
        "e.g. 'FP,SIGFP'. Set when auto-detection fails or picks the wrong array."
    ),
)
@click.option(
    "--mtz_rfree_label",
    type=str,
    default=None,
    help=(
        "R-free flag column name, e.g. 'FreeR_flag'. "
        "Set when auto-detection fails or the MTZ contains multiple flag columns."
    ),
)
# SLURM Resources
@click.option("--partition", type=str, help="SLURM partition to use.")
@click.option("--cpus_per_task", type=int, help="SLURM CPUs per task.")
@click.option("--mem_per_cpu", type=int, help="SLURM memory per CPU.")
@click.option(
    "--num_nodes",
    type=int,
    help="Number of nodes to use for the preprocessing and omission steps.",
)
def generate_params(
    config,
    run_name,
    structure_path,
    reflections_path,
    screening_path,
    work_dir,
    omit_type,
    omit_fraction,
    always_omit,
    iterations,
    seed,
    sqlite_outcomes,
    max_structures,
    screening_chunk_size,
    mtz_f_labels,
    mtz_rfree_label,
    partition,
    cpus_per_task,
    mem_per_cpu,
    num_nodes,
):
    """
    Generate SLURM job files for the Debias pipeline.
    Supports internal defaults, external config files, and CLI flags.
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
    if always_omit is not None:
        overrides.append(f"debias.always_omit={always_omit}")
    if iterations is not None:
        overrides.append(f"debias.iterations={iterations}")
    if seed is not None:
        overrides.append(f"debias.seed={seed}")

    if sqlite_outcomes:
        overrides.append(f"debias.sqlite_outcomes={sqlite_outcomes}")
    if max_structures is not None:
        overrides.append(f"debias.max_structures={max_structures}")
    if screening_chunk_size is not None:
        overrides.append(f"debias.screening_chunk_size={screening_chunk_size}")
    if mtz_f_labels is not None:
        overrides.append(f"debias.mtz_f_labels={mtz_f_labels}")
    if mtz_rfree_label is not None:
        overrides.append(f"debias.mtz_rfree_label={mtz_rfree_label}")

    if partition:
        overrides.append(f"slurm.partition={partition}")
    if cpus_per_task:
        overrides.append(f"slurm.cpus_per_task={cpus_per_task}")
    if mem_per_cpu:
        overrides.append(f"slurm.mem_per_cpu={mem_per_cpu}")
    if num_nodes:
        overrides.append(f"slurm.num_nodes={num_nodes}")

    try:
        cfg = load_debias_config(config_path=config, overrides=overrides)
    except Exception as e:
        click.echo(f"Configuration Error: {e}", err=True)
        return

    generate_slurm_job(cfg)
