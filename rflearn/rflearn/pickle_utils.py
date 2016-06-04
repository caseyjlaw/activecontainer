import pickle
import sys
#import cPickle

def write_pickle(data, f_pkl_out):

    fh_out = open(f_pkl_out,'wb')
    pickle.dump(data, fh_out)
    fh_out.close()


def read_pickle(f_pkl_in):

    try:
        fh_in = open(f_pkl_in,'rb')
    except IOError:
        print "Cannot open %s" % f_pkl_in
        sys.exit(-1)
    data = pickle.load(fh_in)
    fh_in.close()
    return data

def pkl2mat(f_pkl_in, f_mat_out):

    fh_in = open(f_pkl_in, 'rb')
    fh_out = open(f_mat_out, 'w')
    data = pickle.load(fh_in)
    for row in data:
	for elem in row:
	   fh_out.write(str(elem) + ' ')
	fh_out.write('\n')

    fh_in.close()
    fh_out.close()



if __name__== '__main__' :

    pkl2mat('combined_reals_alexandra.pkl','combined_reals_alexandra.mat')
    pkl2mat('bogus_random_15K_alexandra.pkl','bogus_random_15K_alexandra.mat')


