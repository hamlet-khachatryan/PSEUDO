import click
from quantify.api import run_quantification
from quantify.end import run_end, RunConfigError
from quantify.delta import run_delta


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
    "--end",
    is_flag=True,
    default=False,
    help="Also compute END (Electron Number Density) maps on the absolute "
    "e-/A^3 scale, alongside the sigma-scaled outputs.",
)
@click.option(
    "--delta",
    is_flag=True,
    default=False,
    help="Also compute delta density maps (mu - model density) for the "
    "perturbation model. Produces a sigma-scaled delta always, and an "
    "END-scaled delta when --end is also set.",
)
def quantify_cli(input_path, stem, force, k_factor, map_cap, num_processes, end, delta):
    """
    Quantify Omission Ensembles
    """

    try:
        run_quantification(
            input_path, stem, force, k_factor, map_cap, num_processes, end, delta
        )
    except Exception as e:
        click.echo(f"Error: {e}")
        raise e


@click.command(name="end")
@click.option(
    "--input_path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the workspace directory containing a completed STOMP run.",
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
    help="Force re-computation, overwriting existing END outputs.",
)
@click.option(
    "--k_factor",
    "-k",
    default=1.0,
    type=float,
    show_default=True,
    help="K factor of the target quantify_results/k_{k}_cap_{cap}/ directory.",
)
@click.option(
    "--map_cap",
    "-c",
    default=50,
    type=int,
    show_default=True,
    help="Number of maps to use from the ensemble. Uses maps 0 to N-1.",
)
def end_cli(input_path, stem, force, k_factor, map_cap):
    """
    Compute END (Electron Number Density) maps for a completed STOMP run.

    Reads each experiment's persisted run configuration to determine the
    bulk-solvent convention, then computes rho_END^(k), mu_END, sigma_END and
    SNR_END on the absolute e-/A^3 scale WITHOUT re-running STOMP. Outputs are
    written alongside the sigma-scaled maps in quantify_results/k_*_cap_*/.
    """
    try:
        run_end(input_path, stem, force, k_factor, map_cap)
    except RunConfigError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise e


@click.command(name="delta")
@click.option(
    "--input_path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the workspace directory containing a completed STOMP run.",
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
    help="Force re-computation, overwriting existing delta outputs.",
)
@click.option(
    "--k_factor",
    "-k",
    default=1.0,
    type=float,
    show_default=True,
    help="K factor of the target quantify_results/k_{k}_cap_{cap}/ directory.",
)
@click.option(
    "--map_cap",
    "-c",
    default=50,
    type=int,
    show_default=True,
    help="Number of maps to use from the ensemble. Uses maps 0 to N-1.",
)
def delta_cli(input_path, stem, force, k_factor, map_cap):
    """
    Compute delta density maps (mu - model density) for a completed STOMP run.

    Subtracts the density predicted by the perturbation model
    ({stem}_updated) from the STOMP mu map. Writes a sigma-scaled delta from
    {stem}_mean.ccp4, and — where {stem}_end_mean.ccp4 already exists — an
    absolute-scale END delta, into quantify_results/k_*_cap_*/ WITHOUT re-running
    STOMP.
    """
    try:
        run_delta(input_path, stem, force, k_factor, map_cap)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise e
