import numpy as np
from sklearn.metrics import confusion_matrix
import pylab as pl

def spec_features(spec, bin):
    rr = np.array([np.abs(spec[:,i0:i0+bin,0].mean(axis=1)) for i0 in range(0,232,bin)])
    ll = np.array([np.abs(spec[:,i0:i0+bin,1].mean(axis=1)) for i0 in range(0,232,bin)])
    sh = rr.shape
    rr = rr[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    ll = ll[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    specbinned = np.concatenate((rr,ll), axis=1)    
    return np.concatenate( (specbinned.mean(axis=0), specbinned.mean(axis=1)) )

def image_features(im):
    sh = im.shape
    imstamp = im[sh[0]/2-10:sh[0]/2+10, sh[1]/2-10:sh[1]/2+10]
    return np.concatenate( (imstamp.mean(axis=0), imstamp.mean(axis=1)) )


def stat_features(stats):
    stats = np.array(stats)
    return np.array(stats)[[0,4,5,6,7,8]]

def plot_imsp(im, spec, bin):
    pl.figure(figsize=(16,4))
    pl.subplot(1, 2, 1)

    sh = im.shape
    im = im[sh[0]/2-10:sh[0]/2+10, sh[1]/2-10:sh[1]/2+10]
    pl.imshow(im, interpolation='nearest')
    pl.subplot(1, 2, 2)
    rr = np.array([np.abs(spec[:,i0:i0+bin,0].mean(axis=1)) for i0 in range(0,232,bin)])
    ll = np.array([np.abs(spec[:,i0:i0+bin,1].mean(axis=1)) for i0 in range(0,232,bin)])
    sh = rr.shape
    rr = rr[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    ll = ll[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    fill = np.zeros_like(rr)
    rrll = np.concatenate((rr,fill,ll), axis=1)

    pl.imshow(rrll, interpolation='nearest')
    pl.show()

def plot_stamps(im, spec, bin):
    pl.figure(figsize=(16,4))
    pl.subplot(1, 2, 1)
    pl.imshow(im, interpolation='nearest')
    pl.subplot(1, 2, 2) 
    pl.imshow(spec, interpolation='nearest')
    pl.show()

def calc_acc_fpr_fnr(targets, preds):
    cm = confusion_matrix(targets, preds)
    tn, tp, fp, fn = cm[0,0], cm[1,1], cm[0,1], cm[1,0]
    return ((tp + tn) /float(tn + tp + fn + fp), fp / float(fp + tn), fn / float(fn + tp) )

def image_stamp(im):
    sh = im.shape 
    imstamp = im[sh[0]/2-10:sh[0]/2+10, sh[1]/2-10:sh[1]/2+10] 
    return imstamp

def spec_stamp(spec, bin):                                                                                                                                                                        
    rr = np.array([np.abs(spec[:,i0:i0+bin,0].mean(axis=1)) for i0 in range(0,232,bin)])
    ll = np.array([np.abs(spec[:,i0:i0+bin,1].mean(axis=1)) for i0 in range(0,232,bin)])
    sh = rr.shape
    rr = rr[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    ll = ll[0:sh[0], sh[1]/2-10:sh[1]/2+10]
    specbinned = np.concatenate((rr,ll), axis=1)
    fill = np.zeros_like(rr)
    rrll = np.concatenate((rr,fill,ll), axis=1)
    return rrll                                                                                                                     
                 
