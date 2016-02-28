import pickle
import json
import requests
from datetime import datetime, timedelta
from rtpipe.parsecands import read_candidates

def read(candsfile, threshold=0):
  """ Read candsfile and get """

  state = pickle.load(open(candsfile))
  loc, prop = read_candidates(candsfile, snrmin=threshold)
  cands = {(tuple(loc[i]), tuple(prop[i])) for i in range(len(loc))}
  return state, cands


def parse(state, cands):
  """ Get cands info into form ready to post """

  alldata = []

  time = datetime.now()
  date = time
  for key in cands:
    data = {}

    data['obs'] = '14A-425'
    date = date + timedelta(hours=1)  # dummy date for now
    data['@timestamp'] = date.isoformat()+'Z'
    data['scan'] = key[0]
    data['segment'] = key[1]
    data['integration'] = key[2]
    data['dm'] = key[3]
    data['dt'] = key[4]
    value = cands[key]
    data['snr'] = value[0]
    idobj = {}
    idobj['_id'] = data['obs'] + '_' + str(data['scan']) + '_' + str(data['segment'])+ '_' + str(data['integration'])

    alldata.append({"index":idobj})
    alldata.append(data)

  jsonStr = json.dumps(alldata,separators=(',', ':'))
  cleanjson = jsonStr.replace('}},','}}\n').replace('},','}\n').replace(']','').replace('[','')

  return cleanjson


def post(cleanjson, url='http://localhost:9200/realfast/cands/_bulk?'):
  """ Post json to elasticsearch instance """

  r = requests.post(url, data=cleanjson)
  print r
