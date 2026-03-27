import click
from analyse.api import run_analysis
from analyse.screen_report import generate_screen_report


@click.command(name="analyse")
@click.option(
    "--input_path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the workspace directory containing debiased and quantified run(s).",
)
@click.option(
    "--stem",
    "-s",
    default=None,
    help="Explicitly specify the experiment stem. If omitted, the stem is inferred from the processed/ directory.",
)
@click.option(
    "--map_path",
    "-m",
    default=None,
    type=click.Path(exists=True),
    help="Path to a CCP4 map to use instead of the auto-discovered SNR map"
    "If omitted, the SNR map from quantify_results/ is used.",
)
@click.option(
    "--model_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a structure file (.pdb or .cif) to score instead of the "
    "processed model from debias.",
)
@click.option(
    "--k_factor",
    "-k",
    default=1.0,
    type=float,
    show_default=True,
    help="K factor used during quantification run. Used to locate the SNR map "
    "in specific quantify_results/k_{k}_cap_{cap}/.",
)
@click.option(
    "--map_cap",
    "-c",
    default=50,
    type=int,
    show_default=True,
    help="Map cap used during quantification run. Used to locate the SNR map. "
    "Pass 0 to auto-detect the highest available cap.",
)
@click.option(
    "--num_processes",
    "-n",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel worker processes for screening mode ",
)
@click.option(
    "--significance_alpha",
    "-a",
    default=0.05,
    type=float,
    show_default=True,
    help="Significance level for the null-distribution SNR threshold (p value threshold). "
    "The SNR value at p=alpha is used as the MUSE classification threshold. Default p=0.05.",
)
def analyse_cli(input_path, stem, map_path, model_path, k_factor, map_cap,
                num_processes, significance_alpha):
    """
    Run MUSE density-support analysis on debiased, quantified structure(s).
    For each experiment, scores every heavy atom against the STOMP-SNR map,
    and writes to analyse_results/ in each experiment directory:

    \b
        {stem}_atoms.csv      per-atom MUSE scores and diagnostic flags
        {stem}_residues.csv   per-residue MUSEm aggregated scores
        {stem}_summary.json   global statistics (OPIA, counts, etc.)
        {stem}_scored.pdb     structure with MUSE scores in the B-factor column
                              (load in PyMOL and colour by b-factor to visualise)

    In screening mode (multiple experiments), an HTML summary report and
    per-experiment JSON files are also written:

    \b
        index.html                                   interactive screen report
        metadata/<stem>_screen_result.json           per-experiment results
        metadata/screen_summary_<timestamp>.json     run-level summary
    """
    try:
        run_analysis(
            input_path=input_path,
            stem=stem,
            map_path=map_path,
            model_path=model_path,
            k_factor=k_factor,
            map_cap=map_cap if map_cap > 0 else None,
            num_processes=num_processes,
            significance_alpha=significance_alpha,
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise e


@click.command(name="screen-report")
@click.option(
    "--input_path",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Root directory of the PSEUDO screening run (the run_name/ directory).",
)
@click.option(
    "--open_browser",
    is_flag=True,
    default=False,
    help="Open index.html in the default browser after generation.",
)
def screen_report_cli(input_path, open_browser):
    """
    (Re)generate the screen-run HTML/JSON report for a completed screening analysis.

    Reads analyse_results/ from every experiment sub-directory and writes:

    \b
        index.html                                    interactive screen report
        metadata/<stem>_screen_result.json            per-experiment results
        metadata/screen_summary_<timestamp>.json      run-level summary

    The original screening project directory is inferred automatically from
    the common ancestor of input structure paths in sbatch/preprocessing_manifest.txt.
    """
    try:
        generate_screen_report(
            screening_dir=input_path,
            open_browser=open_browser,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
