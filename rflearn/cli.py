import click
import rflearn
import logging
logging.basicConfig()


@click.group('rflearn')
def cli():
    pass


@cli.command()
@click.argument('candsfile')
@click.option('--addscores', help='')
@click.option('--tag', help='')
@click.option('--command', help='')
def readandpush(candsfile, addscores, tag, command):
    """ Candidate pickle file to post to realfast index"""

    rflearn.elastic.readandpush(candsfile, push=True)
