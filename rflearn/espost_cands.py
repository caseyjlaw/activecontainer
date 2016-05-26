import pickle, json, requests, logging
from rtpipe.parsecands import read_candidates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse(candsfile, url='', onlyplots=False):
    """ Get cands info into form ready to post 

    optional url command gives location of candidate plots to filter parsed list
    onlyplots says to filter by plots that exist at url
    """

    loc, prop, state = read_candidates(candsfile, returnstate=True)

    fileroot = state['fileroot']
    scancol = state['featureind'].index('scan')
    segmentcol = state['featureind'].index('segment')
    intcol = state['featureind'].index('int')
    dmcol = state['featureind'].index('dmind')
    dtcol = state['featureind'].index('dtind')
    if 'snr2' in state['features']:
        snrcol = state['features'].index('snr2')
    elif 'snr1' in state['features']:
        snrcol = state['features'].index('snr1')

    alldata = []

    for i, key in enumerate(loc):
        data = {}
        data['obs'] = fileroot
        data['scan'] = key[scancol]
        data['segment'] = key[segmentcol]
        data['integration'] = key[intcol]
        data['dm'] = key[dmcol]
        data['dt'] = key[dtcol]
        value = prop[i]
        data['snr'] = value[snrcol]
 
        uniqueid = '{0}_sc{1}-seg{2}-i{3}-dm{4}-dt{5}'.format(data['obs'], data['scan'], data['segment'], data['integration'], data['dm'], data['dt'])
        data['candidate_png'] = 'cands_{0}.png'.format(uniqueid)
  
        idobj = {}
        idobj['_id'] = uniqueid
        if onlyplots:
            if os.path.exists(os.path.join(url, data['candidate_png'])):
                alldata.append({"index":idobj})
                alldata.append(data)
            else:
                logger.info('Plot not found for {0}. Not including.'.format(data['candidate_png']))

    jsonStr = json.dumps(alldata,separators=(',', ':'))
    cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','')

    return cleanjson


def post(cleanjson, url='http://localhost:9200/realfast/cands/_bulk?'):
    """ Post json to elasticsearch instance """

    r = requests.post(url, data=cleanjson)
    logger.info('Post status: {0}'.format(r))
