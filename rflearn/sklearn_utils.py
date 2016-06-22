# given confusion matrix scores output from the
# classifier, calculate the FPR and FNR

import sys
import numpy as np
import pylab as pl
from scipy import interp
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.externals import joblib

def test_scores(clf, train_dat, train_labels, test_dat):
    scores = clf.fit(train_dat,train_labels).predict_proba(test_dat)
    return scores

def plot_stable_features(X_train,y_train,featnames,**kwargs):
    from sklearn.linear_model import LassoLarsCV,RandomizedLasso

    n_resampling = kwargs.pop('n_resampling',200)
    n_jobs = kwargs.pop('n_jobs',-1)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        # estimate alphas via xvalidation 
        lars_cv = LassoLarsCV(cv=6,n_jobs=n_jobs).fit(X_train,y_train)        
        alphas = np.linspace(lars_cv.alphas_[0], .1 * lars_cv.alphas_[0], 6)

        clf = RandomizedLasso(alpha=alphas, random_state=42, n_jobs=n_jobs,
                              n_resampling=n_resampling)
        clf.fit(X_train,y_train)
        importances = clf.scores_ 
        indices = np.argsort(importances)[::-1]

        pl.bar(range(len(featnames)), importances[indices],
               color="r", align="center")
        pl.xticks(np.arange(len(featnames))+0.5,featnames[indices],
                  rotation=45,horizontalalignment='right')
        pl.xlim(-0.5,len(featnames)-0.5)
        pl.subplots_adjust(bottom=0.2)
        
        pl.ylim(0,np.max(importances)*1.01)
        pl.ylabel('Selection frequency (%) for %d resamplings '%n_resampling)
        pl.title("Stability Selection: Selection Frequencies")

def plot_importances(clf,featnames,outfile,**kwargs):

    pl.figure(figsize=(16,4))

    featnames = np.array(featnames)
    importances = clf.feature_importances_
    imp_std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
    indices = np.argsort(importances)[::-1]

    #for featname in featnames[indices]:
    #    print featname

    trunc_featnames = featnames[indices]
    trunc_featnames = trunc_featnames[0:24]
    trunc_importances = importances[indices]
    trunc_importances = trunc_importances[0:24]
    trunc_imp_std = imp_std[indices]
    trunc_imp_std = trunc_imp_std[0:24]

    pl.bar(range(len(trunc_featnames)), trunc_importances,
           color="r", yerr=trunc_imp_std, align="center")
    pl.xticks(np.arange(len(trunc_featnames))+0.5,trunc_featnames,rotation=45,
              horizontalalignment='right')
    pl.xlim(-0.5,len(trunc_featnames)-0.5)
    pl.ylim(0,np.max(trunc_importances+trunc_imp_std)*1.01)

#    pl.bar(range(len(featnames)), importances[indices],
#           color="r", yerr=imp_std[indices], align="center")
#    pl.xticks(np.arange(len(featnames))+0.5,featnames[indices],rotation=45,
#              horizontalalignment='right')
#    pl.xlim(-0.5,len(featnames)-0.5)
#    pl.ylim(0,np.max(importances+imp_std)*1.01)
    pl.subplots_adjust(bottom=0.2)
    pl.show()
    #pl.savefig(outfile)
    #pl.clf()

#def plot_importances(clf,featnames,outfile,**kwargs):
#
#    importances = clf.feature_importances_
#    imp_std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#                     axis=0)
#    indices = np.argsort(importances)[::-1]
#    
#    pl.bar(range(len(featnames)), importances[indices],
#           color="r", yerr=imp_std[indices], align="center")
#    pl.xticks(np.arange(len(featnames))+0.5,featnames[indices],rotation=45,
#              horizontalalignment='right')
#    pl.xlim(-0.5,len(featnames)-0.5)
#    pl.ylim(0,np.max(importances+imp_std)*1.01)
#    pl.subplots_adjust(bottom=0.2)
#    pl.savefig(outfile)
#    pl.clf()
#
def init_clf(X_train,y_train,trainfn,clf_file,**kwargs):
    from os.path import exists as pathexists

    retrain = kwargs.pop('retrain',False)
    if not pathexists(clf_file) or retrain:
        clf = trainfn(X_train,y_train,**kwargs)
        _ = save_classifier(clf,clf_file)
    else:
        clf = load_classifier(clf_file)
    return clf

def init_random_forest(X_train,y_train,clf_file,**kwargs):
    print 'initializing random forest %s'%clf_file
    nfeat = X_train.shape[1]
    maxfeat = np.unique(range(1,nfeat,5)+[nfeat])
    kwargs['tuned_params'] = [{'max_features':maxfeat}]
    return init_clf(X_train,y_train,train_random_forest,clf_file,**kwargs)

def train_random_forest(X_train,y_train,**kwargs):
    from sklearn.ensemble import ExtraTreesClassifier

    n_estimators = kwargs.pop('n_estimators',300)
    max_features = kwargs.pop('max_features','auto')
    n_jobs       = kwargs.pop('n_jobs',-1)
    verbose      = kwargs.pop('verbose',0)
    tuned_params = kwargs.pop('tuned_params',None)

    # initialize baseline classifier
    clf = ExtraTreesClassifier(n_estimators=n_estimators,random_state=42,
                               n_jobs=n_jobs,verbose=verbose,criterion='gini',
                               max_features=max_features,oob_score=True,
                               bootstrap=True)
    
    if tuned_params is not None: # optimize if desired
        from sklearn.grid_search import GridSearchCV
        cv = GridSearchCV(clf,tuned_params,cv=5,scoring='roc_auc',
                          n_jobs=n_jobs,verbose=verbose,refit=True)
        cv.fit(X_train, y_train)
        clf = cv.best_estimator_
    else: # otherwise train with the specified parameters (no tuning)
        clf.fit(X_train,y_train)

    return clf

def init_svm(X_train,y_train,clf_file,**kwargs):
    C_range = [1,10,100,1000]
    gamma_range = [0.0001,0.1,1,10,1000]
    kwargs['tuned_params'] = [{'kernel':['rbf'],'C':C_range,'gamma':gamma_range}]
    return init_clf(X_train,y_train,train_svm,clf_file,**kwargs)
    
def train_svm(X_train,y_train,**kwargs):
    from sklearn.svm import SVC

    C            = kwargs.pop('C',1.0)
    kernel       = kwargs.pop('kernel','rbf')
    gamma        = kwargs.pop('gamma',0.0)    
    verbose      = kwargs.pop('verbose',0)
    tuned_params = kwargs.pop('tuned_params',None)
    n_jobs       = kwargs.pop('n_jobs',-1)
    
    # initialize baseline classifier
    clf = SVC(C=C,kernel=kernel,gamma=gamma,verbose=verbose,              
              random_state=42,probability=True)
    
    if tuned_params is not None: # optimize if desired
        from sklearn.grid_search import GridSearchCV            
        cv = GridSearchCV(clf,tuned_params,cv=5,scoring='roc_auc',
                          n_jobs=n_jobs,verbose=verbose,refit=True)
        cv.fit(X_train, y_train)
        clf = cv.best_estimator_
    else: # otherwise train with the specified parameters (no tuning)
        clf.fit(X_train,y_train)

    return clf

def save_classifier(clf, filename):
    ret = joblib.dump(clf, filename, compress=9)
    print "saved classifier to file %s"%filename, ret
    return ret

def load_classifier(filename):
    clf = joblib.load(filename)
    print "loaded classifier from file %s"%filename
    return clf
    
def fpr_fnr_from_crossval(clf, data, labels, **kwargs):
    data = np.array(data)
    labels = np.array(labels)    
    
    n_folds = kwargs.pop('n_folds',5)
    cv = StratifiedKFold(labels, n_folds=n_folds)
    scores = []
    for i, (train,test) in enumerate(cv):
        preds = clf.fit(data[train],labels[train]).predict(data[test])
        cmat = confusion_matrix(labels[test],preds)
        scores.append(cmat)

    return get_fpr_fnr(scores)

# def compute_roc(data,labels,num_folds):
#     # CROSS VALIDATION and computing predictive probabilities
#     cv = StratifiedKFold(labels, n_folds=num_folds)
    
#     fprs = []
#     fnrs = []
    
#     for i, (train, test) in enumerate(cv): 
#         # get probabilities from the classifier
#         probs_test = clf.fit(data[train], labels[train]).predict_proba(data[test]) 
#         labels_test = labels[test]
        
#         # sweep a decision threshold through the probabilities
#         for thres in np.linspace(0, 1, num=11):
#             conf_matrix = [][]
            
#             for j in range(len(probs_test)):               
#                 row_probs = probs[j]
#                 label = labels_test[j]
                
#                 if row[1] < thres:
#                     pred_pos += 1
#                 else:
#                     pred_neg += 1
                    
#                 conf_matrix[0][0] = true_neg
#                 conf_matrix[0][1] = false_pos 
#                 conf_matrix[1][0] = false_neg 
#                 conf_matrix[1][1] = true_pos

def balance_classes(y):
    from numpy.random import permutation as randperm
    from numpy import random

    random.seed(1)
    yuniq = np.unique(y)
    ycounts = [np.sum(y==yi) for yi in yuniq]
    ymax = np.max(ycounts)
    balidx = np.array([],int)
    for i,yi in enumerate(yuniq):
        yidx = np.where(y==yi)[0]
        if ycounts[i] < ymax:
            nadd = ymax-ycounts[i]
            perm = np.asarray(randperm(ycounts[i]),int)
            balidx = np.r_[balidx,yidx[perm[:nadd]]]
    return balidx


# subroutine to get FPR and FNR
# from a set of cv_scores
#
def get_fpr_fnr(cv_scores):
    fpr_list = []
    fnr_list = []
    acc_list = []

    for conf_matrix in cv_scores:
        true_neg = conf_matrix[0][0]
        false_pos = conf_matrix[0][1]
        false_neg = conf_matrix[1][0]
        true_pos = conf_matrix[1][1]

        fpr = false_pos / float(false_pos + true_neg)
        fnr = false_neg / float(false_neg + true_pos)
        acc = float(true_pos + true_neg) / np.sum(conf_matrix)
        
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        acc_list.append(acc)

    return [ np.mean(fpr_list), np.mean(fnr_list), np.mean(acc_list) ]


def roc_from_crossval(classifier, X, y, num_folds, balance=False,
                      test_exclude=[], X_test=[], y_test=[], compute_weights=None):

    # updated from: http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py
    uy = np.unique(y)

    # CROSS VALIDATION and computing predictive probabilities
    cv_train = StratifiedKFold(y, n_folds=num_folds)

    cv_test = None
    if len(y_test) != 0:
        cv_test = [(train,test) for (train,test) in
                   StratifiedKFold(y_test, n_folds=num_folds)]
    else:
        X_test,y_test = X,y # note: still using cv_train indices for test samples
    
    ftpr_interp = np.linspace(0, 1, 101)
    roc_curves = []
    mean_fpr, mean_thres_interp2fpr = [], []
    mean_tpr, mean_thres_interp2tpr = [], []

    _1fpr_interp_idx = ftpr_interp==0.01
    fnr1fprs = []
    scores = []
    test_err_idx = []
    roc_label = 'ROC (area = %0.6f)'

    for i, (train, test) in enumerate(cv_train):

        if balance:
            train = np.r_[train, balance_classes(y[train])]

        # use cv_test test indices if X_test, y_test are provided
        if cv_test is not None:
            _,test = cv_test[i]
    
        if len(test_exclude)>0:
            test = np.setdiff1d(test,test_exclude)
            if len(test)==0 or np.sum(y_test[test]==uy[0])==0 or np.sum(y_test[test]==uy[1])==0:
                # all test points for at least one class excluded by test_exclude 
                continue

        if compute_weights is not None:
            weights=compute_weights(X[train],X_test[test])
            classifier.fit(X[train], y[train], sample_weight=weights)
        else:
            classifier.fit(X[train], y[train])
        probas_ = classifier.predict_proba(X_test[test])
        preds_ = classifier.predict(X_test[test])
        cmat = confusion_matrix(y_test[test],preds_)
        scores.append(cmat)
        test_err_idx.append(test[y_test[test]!=preds_])

        fpri, fnri, acci = get_fpr_fnr([cmat])
        print 'Fold %d ntest = %d'%(i,len(test))
        print '\ttrain: %d pos %d neg'%(np.sum(y[train]==uy[1]),np.sum(y[train]==uy[0]))
        print '\ttest: %d pos %d neg'%(np.sum(y_test[test]==uy[1]),np.sum(y_test[test]==uy[0]))
        print '\tfpr =', fpri, ' fnr =', fnri, ' acc=', acci
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test[test], probas_[:, 1])

        #thresholds_aslist, fpr_aslist, tpr_aslist = thresholds.tolist(), fpr.tolist(), tpr.tolist()
        #thresholds_aslist.reverse(), fpr_aslist.reverse(), tpr_aslist.reverse()
        #thresholds_rev, fpr_rev, tpr_rev = np.array(thresholds_aslist), np.array(fpr_aslist), np.array(tpr_aslist)
        roc_curves.append((fpr,tpr,thresholds))

        roc_auc = auc(fpr, tpr)
        #pl.plot(fpr, tpr, lw=1, label='Fold %d %s'%(i,roc_label%roc_auc))

        tpr_interp_to_fpr = interp(ftpr_interp, fpr, tpr)
        thresholds_interp_to_fpr = interp(ftpr_interp, fpr, thresholds)
        thresholds_interp_to_tpr = interp(ftpr_interp, tpr, thresholds)

        # pl.clf()
        # pl.plot(fpr, thresholds, 'bo')
        # pl.plot(ftpr_interp, thresholds_interp_to_fpr, 'rx')
        # pl.plot(tpr, thresholds, 'go')
        # pl.plot(ftpr_interp, thresholds_interp_to_tpr, 'rx')
        # pl.savefig('foo.pdf')
        # pl.clf()

        mean_tpr.append(tpr_interp_to_fpr)
        mean_thres_interp2fpr.append(thresholds_interp_to_fpr)
        mean_thres_interp2tpr.append(thresholds_interp_to_tpr)
        fnr1fpri = 1-tpr_interp_to_fpr[_1fpr_interp_idx][0]
        fnr1fprs.append(fnr1fpri)
        print '\tfpr =', 0.1, ' fnr=', fnr1fpri

    num_good_folds = len(scores)
    if num_good_folds==0:
        print 'error: all folds empty'
        return {}

    mean_thres_interp2tpr = np.mean(np.asarray(mean_thres_interp2tpr),axis=0)
    mean_thres_interp2fpr = np.mean(np.asarray(mean_thres_interp2fpr),axis=0)
    mean_tpr = np.mean(np.asarray(mean_tpr),axis=0)
    mean_fpr = ftpr_interp
    mean_fnr1fpr = np.mean(np.asarray(fnr1fprs))
    std_fnr1fpr = np.std(np.asarray(fnr1fprs))
    print mean_tpr[0]
    #mean_tpr[0] = 0 # I believe this was here to get the plot to start from 0,0

    # pl.clf()
    # pl.plot(mean_thres_interp2fpr, mean_fpr, 'bx')
    # pl.plot(mean_thres_interp2tpr, mean_tpr, 'gx')
    # pl.savefig('foo2.pdf')
    # pl.clf()

    fpr, fnr, acc = get_fpr_fnr(scores)
    print '%d fold average'%(num_good_folds)
    print '\tfpr =', fpr, ' fnr =', fnr, ' acc =', acc
    print 'FPR =',0.01,'FNR =',mean_fnr1fpr, '( std =',std_fnr1fpr,')'
    mean_auc = auc(mean_thres_interp2fpr, mean_thres_interp2tpr)

    return {'mean_fpr':mean_fpr, 'mean_tpr':mean_tpr,
            'mean_thres_interp2fpr': mean_thres_interp2fpr, 'mean_thres_interp2tpr': mean_thres_interp2tpr,
            'mean_auc':mean_auc, 'scores':scores, 'test_err_idx':test_err_idx,
            'mean_fnr1fpr':mean_fnr1fpr, 'std_fnr1fpr':std_fnr1fpr}

def plot_mean_roc(mean_tpr,mean_fpr,label='Mean ROC'):    
    pl.plot(mean_fpr, mean_tpr, '--', label='%s'%(label), lw=4)

    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
