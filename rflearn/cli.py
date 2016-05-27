import click
import rflearn


@click.group('rflearn')
def cli():
    pass


@cli.command()
@click.argument('candsfile')
def postcands(candsfile):
    """ Candidate pickle file to post to realfast index"""

    rflearn.elastic.postcands(candsfile)
