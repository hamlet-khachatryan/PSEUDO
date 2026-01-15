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
@click.option("--structure_path", type=click.Path(exists=True), help="Input PDB path.")
@click.option(
    "--reflections_path", type=click.Path(exists=True), help="Input MTZ path."
)
@click.option(
    "--screening_path",
    type=click.Path(exists=True),
    help="Path to the screening file. CSV with 'PDB' and 'MTZ' column names. SoakDB sql file for the Diamond XChem "
    "Data with structure status label for filtration.",
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
