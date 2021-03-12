import sys
import bz2
import pickle
import _pickle as cPickle
import numpy


# Load any compressed pickle file
def decompress_pickle(filename):
    f = bz2.BZ2File(filename, 'rb')
    data = cPickle.load(f)
    f.close()
    return data


verbose = 0

#filename = 'clean_signals/chb01/chb01_07.edf.pkl.pbz2'

data_pieces = list()

for filename in sys.argv[1:]:
    print('loading', filename, end = ' ')
    signal_dict = decompress_pickle(filename)
    metadata = signal_dict.get('metadata')
    if verbose > 0:
        print(signal_dict.keys())
        print(metadata.keys())


    signal_ids = metadata['channels']
    num_seizures = metadata['seizures']
    episodes = metadata['times']

    if len(episodes) > 0:
        episodes.sort(key = lambda a: a[0], reverse = True)

    if verbose > 0:
        print(signal_ids)
        print(num_seizures)
        print(episodes)

    i0 = 0
    for boundaries in episodes:
        i1 = boundaries[0]
        data_pieces.append(numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T)
        print(f' {i1-i0}', end = ' ')
        i0 = boundaries[1] + 1
    i1 = len(signal_dict[signal_ids[0]])
    data_pieces.append(numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T)
    print(f' {i1-i0}', end = ' ')
    #
    print(' complete!')

for data in data_pieces: print(data.shape)
