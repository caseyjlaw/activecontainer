from setuptools import setup, find_packages

setup(
    name = 'rflearn',
    description = 'rtpipe/realfast classification, active learning, and candidate indexing',
    author = 'Umaa Rebbapragada, Shakeh Khudikyan, Casey Law',
    author_email = 'caseyjlaw@gmail.com',
    version = '0.25',
    url = 'http://github.com/caseyjlaw/rflearn',
    packages = find_packages(),        # get all python scripts in real time
    install_requires=['rtpipe', 'scikit-learn', 'activegit', 'click', 'elasticsearch'],  # will add elasticsearch some day
    entry_points='''
        [console_scripts]
        rflearn=rflearn.cli:cli
'''

)
