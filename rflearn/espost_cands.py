import pickle, json, requests, logging, os
from rtpipe.parsecands import read_candidates

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse(candsfile, plotdir='/users/claw/public_html/plots', onlyplots=True):
    """ Get cands info into form ready to post 

    plotdir command gives location of candidate plots to filter parsed list
    onlyplots says to filter by plots that exist in plotdir
    """

    loc, prop, state = read_candidates(candsfile, returnstate=True)

    fileroot = state['fileroot']

    alldata = []
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
  
        idobj = {}
        idobj['_id'] = uniqueid
        if onlyplots:
            if os.path.exists(os.path.join(plotdir, data['candidate_png'])):
                alldata.append({"index":idobj})
                alldata.append(data)
            else:
                logger.info('Plot not found for {0}. Not including.'.format(data['candidate_png']))

    jsonStr = json.dumps(alldata,separators=(',', ':'))
    cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','') + '\n'

    return cleanjson


def post(cleanjson, url='http://136.152.227.149:9200/realfast/cand/_bulk?'):
    """ Post json to elasticsearch instance """

    r = requests.post(url, data=cleanjson)
    logger.info('Post status: {0}'.format(r))
