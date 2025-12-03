import click
from src.debias.api import generate_slurm_job, load_debias_config


@click.group(name="debias")
def debias_cli():
    """Commands for the Debias/Omit Map module."""
    pass


@debias_cli.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True),
    help="Path to external YAML configuration file.",
)
@click.option("--run-name", help="Run name.")
@click.option("--structure", help="Input PDB path.")
@click.option("--reflections", help="Input MTZ path.")
@click.option(
    "--screening",
    help="Path to the screening file. CSV with 'PDB' and 'MTZ' column names. SoakDB sql file for the Diamond XChem Data with structure status label for filtration.",
)
@click.option("--work-dir", help="Working directory.")
# Omission Parameters
@click.option(
    "--omit-type",
    type=click.Choice(["amino_acids", "atoms"]),
    help="Type of elements to omit.",
)
@click.option(
    "--omit-fraction",
    type=float,
    help="Fraction of structure to omit per iteration.",
)
@click.option("--iterations", type=int, help="Number of omission iterations.")
@click.option("--seed", type=int, help="Random seed for reproducibility.")
# SLURM Resources
@click.option("--partition", help="SLURM partition to use.")
@click.option("--cpus-per-task", type=int, help="SLURM CPUs per task.")
@click.option("--mem-per-cpu", help="SLURM memory per CPU.")
def generate(
    config_file,
    run_name,
    structure,
    reflections,
    screening,
    work_dir,
    omit_type,
    omit_fraction,
    iterations,
    seed,
    partition,
    cpus_per_task,
    mem_per_cpu,
):
    """
    Generate SLURM job files for the Debias pipeline.
    Supports internal defaults, external config files, and CLI flags.
    """

    overrides = []
    if run_name:
        overrides.append(f"debias.run_name={run_name}")
        overrides.append(f"slurm.job_name={run_name}")

    if structure:
        overrides.append(f"debias.structure_path={structure}")
    if reflections:
        overrides.append(f"debias.reflections_path={reflections}")
    if screening:
        overrides.append(f"debias.screening_path={screening}")
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

    if partition:
        overrides.append(f"slurm.partition={partition}")
    if cpus_per_task:
        overrides.append(f"slurm.cpus_per_task={cpus_per_task}")
    if mem_per_cpu:
        overrides.append(f"slurm.mem_per_cpu={mem_per_cpu}")

    try:
        cfg = load_debias_config(config_path=config_file, overrides=overrides)
    except Exception as e:
        click.echo(f"Configuration Error: {e}", err=True)
        return

    generate_slurm_job(cfg)
