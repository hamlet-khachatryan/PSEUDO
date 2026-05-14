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
    default=1.0,
    type=float,
    show_default=True,
    help="Radius multiplier coefficient (K) for atom ownership.",
)
@click.option(
    "--map_cap",
    "-c",
    default=50,
    type=int,
    show_default=True,
    help="Number of maps to use from the ensemble. Uses maps 0 to N-1.",
)
@click.option(
    "--num_processes",
    "-n",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel processes for screening mode (multiple experiments).",
)
@click.option(
    "--null_fit_method",
    "-m",
    default="truncated",
    type=click.Choice(["full", "truncated"], case_sensitive=True),
    show_default=True,
    help=(
        "Null-distribution fitting method. 'truncated' uses truncated MLE on the left half of the SNR distribution "
        "to avoid signal contamination from ordered waters.'full' uses unrestricted t.fit on "
        "all background samples."
    ),
)
def quantify_cli(input_path, stem, force, k_factor, map_cap, num_processes, null_fit_method):
    """
    Quantify Omission Ensembles
    """

    try:
        run_quantification(input_path, stem, force, k_factor, map_cap, num_processes, null_fit_method)
    except Exception as e:
        click.echo(f"Error: {e}")
        raise e
