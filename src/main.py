import click
from debias.cli import debias_cli
from quantify.cli import quantify_cli
from analyse.cli import analyse_cli, screen_report_cli, water_sites_cli


@click.group()
def cli():
    """
    PSEUDO main function for CLI entrypoint
    """
    pass


cli.add_command(debias_cli)
cli.add_command(quantify_cli)
cli.add_command(analyse_cli)
cli.add_command(screen_report_cli)
cli.add_command(water_sites_cli)

if __name__ == "__main__":
    cli()
