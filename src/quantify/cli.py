import click
from quantify.api import run_quantification


@click.command(name="quantify")
@click.option(
    "--input_path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the workspace directory containing debiased run(s)",
)
@click.option(
    "--stem",
    "-s",
    default=None,
    help="Explicitly specify the experiment stem.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-computation of all steps, overwriting existing results.",
)
@click.option(
    "--k_factor",
    "-k",
    default=1.5,
    type=float,
    show_default=True,
    help="Radius multiplier coefficient (K) for atom ownership.",
)
@click.option(
    "--map_cap",
    "-c",
    default=None,
    type=int,
    show_default=True,
    help="Number of maps to use from the ensemble (e.g., 90). Uses maps 0 to N-1.",
)
def quantify_cli(input_path, stem, force, k_factor, map_cap):
    """
    Quantify Omission Ensembles.
    Generates Signal, Noise, and SNR maps using robust matrix subtraction.
    """
    try:
        run_quantification(input_path, stem, force, k_factor, map_cap)
    except Exception as e:
        click.echo(f"Error: {e}")
        raise e
