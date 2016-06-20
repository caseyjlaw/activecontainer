import json, requests, os.path, logging
from shutil import copy
from rtpipe.parsecands import read_candidates
from elasticsearch import Elasticsearch
from rflearn.features import stat_features
from rflearn.classify import calcscores
from IPython.display import Image
from IPython.core.display import HTML 
from numpy import isnan, nan_to_num

logging.basicConfig()
es = Elasticsearch(['realfast.berkeley.edu:9200'])  # index in berkeley cloud


def readandpush(candsfile, push=True, addscores=True, tag=None, command='index'):
    """ Read, classify, and push candidates to realfast index.

    Optionally push to index with scores. 
    Optionally can add string to 'tag' field.
    """

    datalist = readcandsfile(candsfile, tag=tag)  # fill plotdir before this
    if classify:
        scores = classify(datalist)

    if addscores:
        for i in range(len(datalist)):
            datalist[i]['rbscore'] = scores[i]

    if push:
        res = pushdata(datalist, command=command)
        logging.info('Post status: {0}'.format(res))
    else:
        return datalist


def readcandsfile(candsfile, plotdir='/users/claw/public_html/plots', tag=None, copyplots=True):
    """ Read candidates from pickle file and format as list of dictionaries

    plotdir is path to png plot files which are required in order to keep in datalist
    optionally copies png files into plotdir
    """

    if tag:
        assert isintance(tag, str)

    loc, prop, state = read_candidates(candsfile, returnstate=True)

    fileroot = state['fileroot']
    if plotdir:
        logging.info('Filtering data based on presence of png files in {0}'.format(plotdir))
    else:
        logging.info('Appending all data to datalist.')

    datalist = []
    for i in range(len(loc)):
        data = {}
        data['obs'] = fileroot

        for feat in state['featureind']:
            col = state['featureind'].index(feat)
            data[feat] = loc[i][col]

        for feat in state['features']:
            col = state['features'].index(feat)
            if isnan(prop[i][col]):
                data[feat] = nan_to_num(prop[i][col])
            else:
                data[feat] = prop[i][col]

        uniqueid = dataid(data)
        data['candidate_png'] = 'cands_{0}.png'.format(uniqueid)
        data['labeled'] = '0'
        if tag:
            data['tag'] = tag
        else:
            data['tag'] = ''

        # copy plot over and add path to datalist
        if plotdir:
            if copyplots and os.path.exists(data['candidate_png']) and not os.path.exists(os.path.join(plotdir, data['candidate_png'])):
                copy(data['candidate_png'], plotdir)
            if os.path.exists(os.path.join(plotdir, data['candidate_png'])):
                datalist.append(data)
        else:
            datalist.append(data)

    return datalist


def indextodatalist(unlabeled=True):
    """ Get all from index and return datalist """

    # add logic to filter for certain tag (e.g., labelled) or presence of certain field (e.g, rbscore)

#    fields = ','.join(features + featureind + ['obs', 'candidate_png'])

    count = es.count()['count']
    if unlabeled:
        res = es.search(index='realfast', doc_type='cand', body={"query": {"term": {"labeled": "0"}}, "size": count})
    else:
        res = es.search(index='realfast', doc_type='cand', body={"query": {"match_all": {}}, "size": count})

    return [hit['_source'] for hit in res['hits']['hits']]


def restorecands(datalist, features=['snr1', 'immax1', 'l1', 'm1', 'specstd', 'specskew', 'speckurtosis', 'imskew', 'imkurtosis'],
                featureind=['scan', 'segment', 'int', 'dmind', 'dtind', 'beamnum']):
    """ Take list of dicts and forms as list of lists in rtpipe standard order

    Order of features and featureind lists is important.
    """

    obslist = []
    loclist = []
    proplist = []

    for data in datalist:
        # build features per data dict
        loc = []
        prop = []
        for fi in featureind:
            loc.append(data[fi])
        for fe in features:
            prop.append(data[fe])

        # append data
        obslist.append(data['obs'])
        loclist.append(tuple(loc))
        proplist.append(tuple(prop))

    return obslist, loclist, proplist


def classify(datalist, agpath='/users/claw/code/alnotebook'):
    """ Applies activegit repo classifier to datalist """

    obslist, loc, prop = restorecands(datalist)
    statfeats = stat_features(prop)
    scores = calcscores(statfeats, agpath=agpath)
    return scores


def pushdata(datalist, index='realfast', doc_type='cand', command='index'):
    """ Pushes list of data to index

    command can be 'index' or 'delete' (update by indexing with same key)
    """

    status = []
    for data in datalist:
        uniqueid = dataid(data)

        if command == 'index':
            res = es.index(index=index, doc_type=doc_type, id=uniqueid, body=data)
        elif command == 'delete':
            res = es.delete(index=index, doc_type=doc_type, id=uniqueid)

        status.append(res['_shards']['successful'])

    return status


def dataid(data):
    """ Returns id string for given data dict """

    return '{0}_sc{1}-seg{2}-i{3}-dm{4}-dt{5}'.format(data['obs'], data['scan'], data['segment'], data['int'], data['dmind'], data['dtind'])


def getids():
    """ Gets candidates from realfast index and returns them as list """

    count = es.count()['count']
    res = es.search(index='realfast', doc_type='cand', fields=['_id'], body={"query": {"match_all": {}}, "size": count})
    return [hit['_id'] for hit in res['hits']['hits']]


def postjson(cleanjson, url='http://realfast.berkeley.edu:9200/realfast/cand/_bulk?'):
    """ **Deprecated** Post json to elasticsearch instance """

#    jsonStr = json.dumps(postdata,separators=(',', ':'))
#    cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','') + '\n'
#    return cleanjson

    r = requests.post(url, data=cleanjson)
    logging.info('Post status: {0}'.format(r))
