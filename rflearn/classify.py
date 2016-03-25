import os.path
import logging
from rflearn import activegit
from sklearn.externals import joblib # for loading classifier

logging.basicConfig(level=logging.INFO)

# usage
#loc_stats, prop_stats = read_candidates(f_in)
#prop_stats = np.array(prop_stats)
#feats = stat_features(prop_stats)
#scores = classify(feats, rbversion)

def classify(feats, rbversion, njobs=1, verbose=0, agpath=None):
    """ Given feature array and classifier version, return a score per feature """

    # validate RB version
    clfpkl = validate_rbversion(agpath, rbversion)

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
