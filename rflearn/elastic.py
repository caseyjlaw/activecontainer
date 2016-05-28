import json, requests, os
from rtpipe.parsecands import read_candidates
from elasticsearch import Elasticsearch
import activegit
from rflearn.features import stat_features
from rflearn.classify import calcscores

es = Elasticsearch(['136.152.227.149:9200'])  # index on berkeley macbook

def readandpush(candsfile):
    """ Read and push candidates to realfast index """

    datalist = readcandsfile(candsfile)
    res = pushdata(datalist)
    return res


def readcandsfile(candsfile, plotdir='/users/claw/public_html/plots'):
    """ Read candidates from pickle file and format as list of dictionaries

    plotdir is path to png plot files which are required in order to keep in datalist
    """

    loc, prop, state = read_candidates(candsfile, returnstate=True)

    fileroot = state['fileroot']
    if plotdir:
        print('Filtering data based on presence of png files in {0}'.format(plotdir))
    else:
        print('Appending all data to datalist.')

    datalist = []
    for i in range(len(loc)):
        data = {}
        data['obs'] = fileroot

        for feat in state['featureind']:
            col = state['featureind'].index(feat)
            data[feat] = loc[i][col]

        for feat in state['features']:
            col = state['features'].index(feat)
            data[feat] = prop[i][col]

        uniqueid = dataid(data)
        data['candidate_png'] = 'cands_{0}.png'.format(uniqueid)

        if plotdir:
            if os.path.exists(os.path.join(plotdir, data['candidate_png'])):
                datalist.append(data)
        else:
            datalist.append(data)

    return datalist


def restorecands(datalist, features=['snr1', 'immax1', 'l1', 'm1', 'specstd', 'specskew', 'speckurtosis', 'imskew', 'imkurtosis'],
                featureind=['scan', 'segment', 'int', 'dmind', 'dtind', 'beamnum']):
    """ Take list of dicts and forms as list of lists in rtpipe standard order """

    keylist = []
    featlist = []
    for data in datalist:
        key = []
        feat = []

        for fi in featureind:
            key.append(data[fi])

        for fe in features:
            feat.append(data[fe])

        keylist.append(tuple(key))
        featlist.append(tuple(feat))

    return (keylist, featlist)


def classify(datalist, agpath='/users/claw/code/alnotebook'):
    """ Applies activegit repo classifier to datalist """

    keys, feats = restorecands(datalist)
    statfeats = stat_features(feats)
    scores = calcscores(statfeats, agpath=agpath)
    return scores


def pushdata(datalist, index='realfast', doc_type='cand', command='index'):
    """ Pushes list of data to index

    command can be 'index', 'update', 'delete'
    """

    status = []
    for data in datalist:
        uniqueid = dataid(data)

        if command == 'index':
            res = es.index(index=index, doc_type=doc_type, id=uniqueid, body=data)
        elif command == 'delete':
            res = es.delete(index=index, doc_type=doc_type, id=uniqueid)
        elif command == 'update':
            res = es.update(index=index, doc_type=doc_type, id=uniqueid, body=data)

        status.append(res)

    return status


def dataid(data):
    """ Returns id string for given data dict """

    return '{0}_sc{1}-seg{2}-i{3}-dm{4}-dt{5}'.format(data['obs'], data['scan'], data['segment'], data['int'], data['dmind'], data['dtind'])


def getids():
    """ Gets candidates from realfast index and returns them as list """

    res = es.search(index='realfast', doc_type='cand', fields=['_id'], body={"query": {"match_all": {}}, "size": 10000})
    return [hit['_id'] for hit in res['hits']['hits']]


def addfield(datalist):
    """ """

    pass


def postjson(cleanjson, url='http://136.152.227.149:9200/realfast/cand/_bulk?'):
    """ **Deprecated** Post json to elasticsearch instance """

#    jsonStr = json.dumps(postdata,separators=(',', ':'))
#    cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','') + '\n'
#    return cleanjson

    r = requests.post(url, data=cleanjson)
    print('Post status: {0}'.format(r))
