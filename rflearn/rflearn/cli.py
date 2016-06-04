import click
import rflearn
import logging
logger = logging.getLogger(__name__)


@click.group('rflearn')
def cli():
    pass


@cli.command()
@click.argument('candsfile')
@click.option('--addscores', help='Classify candidates and add to index', default=True)
@click.option('--tag', help='', default=None)
@click.option('--command', help='Command to use when posting to index (index or delete)', default='index')
def readandpush(candsfile, addscores, tag, command):
    """ Candidate pickle file to post to realfast index"""

    rflearn.elastic.readandpush(candsfile, push=True, addscores=addscores, tag=tag, command=command)
