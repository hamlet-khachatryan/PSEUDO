import click
from debias.cli import debias_cli
# from src.analysis.cli import analysis_cli


@click.group()
def cli():
    """PSEUDO main function for CLI entrypoint."""
    pass


cli.add_command(debias_cli)

if __name__ == "__main__":
    cli()
