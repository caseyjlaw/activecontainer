import json, requests, os
from rtpipe.parsecands import read_candidates


def postcands(candsfile):
    """ Parse and post candidates to realfast index """

    datalist = readcands(candsfile)
    cleanjson = datatojson(datalist)
    postjson(cleanjson)


def readcands(candsfile):
    """ Read candidates from pickle file and format as dictionary

    """

    loc, prop, state = read_candidates(candsfile, returnstate=True)

    fileroot = state['fileroot']

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

        uniqueid = '{0}_sc{1}-seg{2}-i{3}-dm{4}-dt{5}'.format(data['obs'], data['scan'], data['segment'], data['int'], data['dmind'], data['dtind'])
        data['candidate_png'] = 'cands_{0}.png'.format(uniqueid)
        datalist.append(data)

    return datalist


def getcands():
    """ Gets candidates from realfast index and returns them as dict """

    pass


def addfield(datalist):
    """ """

    pass


def datatojson(datalist, plotdir='/users/claw/public_html/plots', onlyplots=True):
    """ Converts list of data dicts to json.

    plotdir command gives location of candidate plots to filter parsed list
    onlyplots says to filter by plots that exist in plotdir
    """

    postdata = []
    for data in datalist:
        uniqueid = '{0}_sc{1}-seg{2}-i{3}-dm{4}-dt{5}'.format(data['obs'], data['scan'], data['segment'], data['int'], data['dmind'], data['dtind'])
        idobj = {}
        idobj['_id'] = uniqueid
        if onlyplots:
            if os.path.exists(os.path.join(plotdir, data['candidate_png'])):
                postdata.append({"index":idobj})
                postdata.append(data)
        else:
            print('Plot not found for {0}. Not including.'.format(data['candidate_png']))

    jsonStr = json.dumps(postdata,separators=(',', ':'))
    cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','') + '\n'

    return cleanjson


def postjson(cleanjson, url='http://136.152.227.149:9200/realfast/cand/_bulk?'):
    """ Post json to elasticsearch instance """

    r = requests.post(url, data=cleanjson)
    print('Post status: {0}'.format(r))
