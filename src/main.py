import click
from debias.cli import debias_cli
from quantify.cli import quantify_cli


@click.group()
def cli():
    """PSEUDO main function for CLI entrypoint."""
    pass


cli.add_command(debias_cli)
cli.add_command(quantify_cli)

if __name__ == "__main__":
    cli()
