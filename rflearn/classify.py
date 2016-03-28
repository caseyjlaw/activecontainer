import os.path
import logging
from rflearn import activegit
from sklearn.externals import joblib # for loading classifier
from numpy import nan_to_num

logging.basicConfig(level=logging.INFO)


def classify(feats, agpath, rbversion=None, njobs=1, verbose=0):
    """ Given feature array and classifier version, return a score per feature

    feats is a numpy array and number of features must match that of rbversion classifier.
    rbversion is a string used to tag version in activegit
    """

    # validate RB version
#    clfpkl = validate_rbversion(agpath, rbversion)
# need to integrate this with new default, which is to check out master by default

    # load classifier and update classifier parameters according to user input
    try:
        clf = load_classifier(clfpkl)
        clf.n_jobs  = njobs
        clf.verbose = verbose
        logging.info('generating predictions for %d samples...'% feats.shape[0])
        scores = clf.predict_proba(feats)[:,1]
    except:
        print "ERROR running the classifier"
        raise

    logging.info('classified predictions done.')

    return scores


def load_classifier(clfpkl):
    """ Loads a pre-trained classifier from file """

    clf = joblib.load(clfpkl)
    logging.info("loaded classifier from file {0}".format(clfpkl))
    return clf


def validate_rbversion(agpath, rbversion):
    """ Confirm that version is available and return path to classifier """

    ag = activegit.ActiveGit(agpath)
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
