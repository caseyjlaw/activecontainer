import os.path, logging
import activegit
from numpy import nan_to_num

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calcscores(feats, agpath, rbversion=None, njobs=1, verbose=0):
    """ Given feature array and classifier version, return a score per feature

    feats is a numpy array and number of features must match that of rbversion classifier.
    rbversion is a string used to tag version in activegit
    """

    # validate RB version
    ag = activegit.ActiveGit(agpath)
    logger.info(ag.versions)
    if rbversion:
        ag.set_version(rbversion)
    logger.info(ag.version)
    clf = ag.read_classifier()

    logger.info('Generating predictions for {0} samples...'.format(feats.shape[0]))
    scores = clf.predict_proba(feats)[:,1]

    return scores


def load_classifier(clfpkl):
    """ Loads a pre-trained classifier from file """

    from sklearn.externals import joblib # for loading classifier

    clf = joblib.load(clfpkl)
    logger.info("loaded classifier from file {0}".format(clfpkl))
    return clf


def validate_rbversion(agpath, rbversion=None):
    """ Confirm that version is available and return path to classifier """

    ag = activegit.ActiveGit(agpath)
    if rbversion:
        assert rbversion in ag.versions
        ag.set_version(rbversion)

    clfpkl = os.path.join(ag.repopath, "classifier.pkl")
    if not os.path.isfile(clfpkl):
        raise IOError("classifier pkl file {0} not found".format(clfpkl))

    return clfpkl


def get_stat_feats(props, d=None, inds=None):
    """ Pulls clean set of stat feats from properties saved in cands pkl file """

    if d:
        print('Parsing state dictionary for cols')
        if 'snr2' in d['features']:
            snrcol = d['features'].index('snr2')
        elif 'snr1' in d['features']:
            snrcol = d['features'].index('snr1')
        specstd = d['features'].index('specstd')
        specskew = d['features'].index('specskew')
        speckurtosis = d['features'].index('speckurtosis')
        imskew = d['features'].index('imskew')
        imkurtosis = d['features'].index('imkurtosis')
        inds = [snrcol, specstd, specskew, speckurtosis, imskew, imkurtosis]
    elif inds:
        print('Using provided inds {0}'.format(inds))
    else:
        print('No inds or d provided.')
        return

    return nan_to_num(props)[:, inds]
