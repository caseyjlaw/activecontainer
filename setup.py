from setuptools import setup, find_packages

setup(
    name = 'rflearn',
    description = 'platform for realfast classification, active learning, and candidate indexing',
    author = 'Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '0.1',
    url = 'http://github.com/caseyjlaw/rflearn',
    packages = find_packages(),        # get all python scripts in real time
    dependency_links = ['http://github.com/caseyjlaw/rtpipe'],
    install_requires=['scikit-learn'],  # will add elasticsearch some day
)
