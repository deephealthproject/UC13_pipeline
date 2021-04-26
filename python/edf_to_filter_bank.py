import sys
import numpy
import time

import multiprocessing as mp

import Preprocessor as preprocessor

from data_utils import load_file, compress_to_pickle

class MySignal:
    def __init__(self, x):
        self.data = x
        self.n_samples = len(x)

'''
an example to run this script:

    find ../clean_signals/chb01/ -type f | python python/edf_to_filter_bank.py

'''

file = sys.stdin

edf_filenames = list()
for line in file:
    input_filename = line.strip()
    edf_filenames.append(input_filename)


def process_file(input_filename):
    print('processing ... ', input_filename, flush = True)
    #return '%18.6f %s' % (time.time(), input_filename)

    data_pieces = load_file(input_filename,
                            exclude_seizures = False,
                            do_preemphasis = False,
                            separate_seizures = False,
                            verbose = 0)

    signals = data_pieces[0][0]
    n_channels = min(signals.shape[1], 23) # warning: the limit to 23 channels is set manually

    preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                subsampling_period = 2000, # in ms
                                                window_length = 4000, # in ms
                                                fb_length = 20, # number of filters
                                                use_mel_scale = False,
                                                max_freq_for_filters = 70)
                    for _ in range(n_channels)]

    fbank = list()
    time_domain_statistics = list()
    for ch in range(n_channels):
        obj = MySignal(signals[:, ch])
        # a value of 0.95 for Preemphasis is used to pre-process audio signals in
        # speech recognizers, however, in the case of EEG we are interested in low
        # frequencies starting at 0.5 Hz, so using a value of 0.5 for Preemphasis
        # helps to remove the DC component while keeping the energy at low frequencies
        preprocessors[ch].preemphasis_alpha = 0.50
        preemphasis, spectrogram, fb, fb_choi, mfcc = preprocessors[ch].preprocess_an_utterance(obj, verbose = 0)

        fbank.append(fb)

        mss = preprocessor.MySignalStats(preemphasis)
        time_domain_statistics.append(mss.time_domain_statistics, window_length = 4 * 256, subsampling_period = 2 * 256) # 4 seconds window every 2 secons

    X = numpy.zeros([len(fbank[0]), len(fbank), fbank[0].shape[1]])
    S = numpy.zeros([len(time_domain_statistics[0]), len(time_domain_statistics), time_domain_statistics[0].shape[1]])
    for i in range(len(fbank)):
        X[:, i, :] = fbank[i][:, :]
    for i in range(len(time_domain_statistics)):
        S[:, i, :] = time_domain_statistics[i][:, :]

    output_filename = input_filename.replace('.edf.', '.fbank.')
    compress_to_pickle(output_filename, X)
    output_filename = input_filename.replace('.edf.', '.td_stats.')
    compress_to_pickle(output_filename, S)
    return '%18.6f %s' % (time.time(), input_filename)

with mp.Pool(processes = mp.cpu_count()) as pool:
    #pool_output = pool.map(my_func, edf_filenames)
    pool_output = pool.map(process_file, edf_filenames)

for s in pool_output:
    print(s)
