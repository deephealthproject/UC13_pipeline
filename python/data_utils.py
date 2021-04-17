import os
import sys
import bz2
import pickle
import _pickle as cPickle
import numpy

from random import shuffle

from multiprocessing import Pool, TimeoutError


def decompress_pickle(filename):
    '''
    Load any compressed pickle file

    :param str filename:
        Name of the file from which to load data.

    :return: The contents of the file.
    '''
    f = bz2.BZ2File(filename, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def compress_to_pickle(filename, obj):
    '''
    Save a Python object into a compressed pickle file
    '''
    if not filename.endswith('.pbz2'):
        filename += '.pbz2'
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(obj, f)



def load_data_one_patient(base_dir = '.',
                            n_processes = 8,
                            verbose = 0,
                            exclude_seizures = False,
                            do_preemphasis = False):
    '''
    Load data from one directory asuming all files belong to the same patient.

    :param str base_dir:
        Path to the directory where the files with extension ".pkl.pbz2" are located.

    :param int n_processes:
        Number of jobs/threads to launch in parallel to load data from files.
        Each thread will load one file at a time.

    :param boolean exclude_seizures:
        To indicate if subsequences corresponding to seizures must be skipped.

    :param boolean do_preemphasis:
        To indicate if FIR preemphasis filter must be applied.

    :param int verbose:
        Level of verbosity.

    :return: A list with the sequences of vectors with channels
             from all the files in the directories provided.
             Each element in the list correspond to a change
             in the label, no seizures imply a single sequence.
    '''

    data_pieces = list()

    filenames = list()
    for filename in os.listdir(base_dir):
        if filename.endswith(".pkl.pbz2"):
            filenames.append(base_dir + '/' + filename)

    return load_files(filenames,
                        n_processes = n_processes,
                        verbose = verbose,
                        exclude_seizures = exclude_seizures,
                        do_preemphasis = do_preemphasis)

def load_files(filename_list,
                n_processes = 8,
                verbose = 0,
                exclude_seizures = False,
                do_preemphasis = False):
    '''
    Loads all the files provided in the parameter **filename_list**

    :param list filename_list: List with the names of files from which to load data.

    :param int n_processes:
        Number of jobs/threads to launch in parallel to load data from files.
        Each thread will load one file at a time.

    :param boolean exclude_seizures:
        To indicate if subsequences corresponding to seizures must be skipped.

    :param boolean do_preemphasis:
        To indicate if FIR preemphasis filter must be applied.

    :param int verbose:
        Level of verbosity.

    :return: A list with the sequences of vectors with channels.
             Each element in the list correspond to a change in
             the label, no seizures imply a single sequence.
    '''

    with Pool(processes = n_processes) as pool:

        pool_output = pool.starmap(load_file, zip(filename_list,
                                                    [exclude_seizures] * len(filename_list),
                                                    [do_preemphasis] * len(filename_list),
                                                    [verbose] * len(filename_list)))

    data_pieces = list()
    for l in pool_output:
        for d in l:
            data_pieces.append(d)

    return data_pieces



def load_file(filename, exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = True,
                        verbose = 0):
    '''
    Loads the contents of one file.

    :param str filename:
        Name of the file from which to load data.

    :param boolean exclude_seizures:
        To indicate whether subsequences corresponding to seizures must be skipped.

    :param boolean do_preemphasis:
        To indicate whether FIR preemphasis filter must be applied.

    :param boolean separate_seizures:
        To indicate whether split the signal into pieces to isolate
        subsequences corresponding to seizures.

    :param int verbose:
        Level of verbosity.

    :return: A list with the sequences of vectors with channels.
             Each element in the list correspond to a change in
             the label, no seizures imply a single sequence.
    '''
    #
    if verbose > 0:
        print('loading', filename,
                'excluding seizures:',exclude_seizures,
                'applying a preemphasis filter:', do_preemphasis)
    #
    signal_dict = decompress_pickle(filename)
    metadata = signal_dict.get('metadata')
    if verbose > 1:
        print(signal_dict.keys())
        print(metadata.keys())

    signal_ids = metadata['channels']
    num_seizures = metadata['seizures']
    episodes = metadata['times']

    if len(episodes) > 0:
        episodes.sort(key = lambda a: a[0], reverse = True)

    if verbose > 1:
        print(signal_ids)
        print(num_seizures)
        print(episodes)

    if do_preemphasis:
        alpha = 0.98
        for key in signal_ids:
            x = signal_dict[key].copy()
            '''
            previous_value = x[0]
            for i in range(1, len(x)):
                temp = x[i]
                x[i] = x[i] - alpha * previous_value
                previous_value = temp
            '''
            signal_dict[key][1:] = x[1:] - alpha * x[:-1]
            signal_dict[key][0] = 0

    data_pieces = list()
    if separate_seizures:
        label = 0 # no seizure, set to 1 for seizures
        i0 = 0
        for boundaries in episodes:
            i1 = boundaries[0]
            _signal = numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T
            _label = label
            label = (label + 1) % 2
            if label == 0 or not exclude_seizures:
                data_pieces.append((_signal, _label))
            i0 = boundaries[1] + 1
        i1 = len(signal_dict[signal_ids[0]])
        _signal = numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T
        _label = label
        if label == 0 or not exclude_seizures:
            data_pieces.append((_signal, _label))
    else:
        _signal = numpy.array([signal_dict[signal_id][:] for signal_id in signal_ids]).T
        data_pieces.append((_signal, 0))
    #
    return data_pieces


class DataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, base_dirs = ['.'], batch_size = 20, window_length = 256, shift = 128,
                 do_shuffle = False,
                 do_standard_scaling = True,
                 do_preemphasis = False,
                 n_processes = 4,
                 exclude_seizures = False,
                 in_training_mode = False):
        '''

            Constructor to create objects of the class **DataGenerator** and loads all the data.
            Future implementations will pave the way to load data from files on-the-fly in order to allow work
            with large enough datasets.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list base_dirs:
                List of directories from which to load data files.

            :param int batch_size:
                Number of samples per batch.

            :param int window_length:
                Size in vectors of channels in the signal to compose a
                single sample.

            :param int shift:
                Number of vectors of channels from the signal to shift from
                one sample to the next one in the sequence.

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale each channel to zero
                mean and unit variance.

            :param boolean do_preemphasis:
                Flag to indicate whether to apply a preemphasis FIR filter
                to each signal/channel.

            :param int n_processes:
                Number of threads to use for loading data from files.

            :param boolean exclude_seizures:
                Flag to indicate whether to exclude the records with seizures.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

        '''
        #
        self.batch_size = batch_size
        self.window_length = window_length
        self.shift = shift
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.do_preemphasis = do_preemphasis
        self.exclude_seizures = exclude_seizures
        self.in_training_mode = in_training_mode
        #
        self.num_batches = 0
        self.num_samples = 0
        self.num_signal_vectors = 0
        #
        self.indices = list()
        self.data_pieces = list()
        _input_shape = None
        for base_dir in base_dirs:
            d_p = load_data_one_patient(base_dir,
                                        n_processes = n_processes,
                                        verbose = 1,
                                        exclude_seizures = self.exclude_seizures,
                                        do_preemphasis = do_preemphasis)
            for p, label in d_p:
                if _input_shape is None:
                    _input_shape = p.shape[1:]
                self.num_signal_vectors += len(p)
                self.data_pieces.append((p, label))
                for i in numpy.arange(0, len(p) - self.window_length, step = self.shift):
                    self.indices.append(((p, label), i)) # we assume Python is not making copies of each data piece
                    self.num_samples += 1
            #
        #
        self.input_shape = (window_length,) + _input_shape
        #
        self.mean = numpy.zeros(self.input_shape[-1])
        self.std  = numpy.ones(self.input_shape[-1])
        #
        if self.do_standard_scaling:
            if self.in_training_mode:
                means = []
                counts = []
                stddevs = []
                for p, label in self.data_pieces:
                    means.append(p.mean(axis = 0))
                    counts.append(len(p))
                    stddevs.append(p.std(axis = 0))
                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)
                #
                array = numpy.array([self.mean, self.std])
                numpy.save('models/statistics.npy', array)
            else:
                array = numpy.load('models/statistics.npy')
                self.mean = array[0]
                self.std = array[1]
        #
        self.on_epoch_end()

    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        return self.num_batches

    def on_epoch_end(self):
        '''
        Performs the predefined actions after an epoch is complete.
        Currently only shuffles data if such option was enabled in the constructor.

        :param self: Reference to the current object.
        '''
        #
        self.num_batches = self.num_samples // self.batch_size

        if self.do_shuffle:
            shuffle(self.indices)
        #

    def __getitem__(self, batch_index : int):
        '''
        Returns the batch of samples specified by the index.

        :param self: Reference to the current object.

        :param int batch_index: Index of the batch to retrieve.

        :return: A tuple with two objects, a batch of samples and the corresponding labels.
        '''
        #
        X = list()
        Y = list()
        #
        for sample in range(self.batch_size):
            (data, label), index = self.indices[batch_index * self.batch_size + sample]

            X.append(data[index : index + self.window_length])
            Y.append(label)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)
        #
        if self.do_standard_scaling:
            X = (X - self.mean) / self.std
        #
        if X.min() < -50. or X.max() > 50.:
            print("#  ", file = sys.stderr)
            print("#  WARNING: too large values after scaling while getting batch %d" % batch_index, file = sys.stderr)
            print("#  min value = %g" % X.min(), file = sys.stderr)
            print("#  max value = %g" % X.max(), file = sys.stderr)
            X = numpy.minimum(X,  50.)
            X = numpy.maximum(X, -50.)
            print("#  values of all the samples in this batch clipped, current limits are [%f, %f]" % (X.min(), X.max()), file = sys.stderr)
            print("#  ", file = sys.stderr)
        #
        return (X, Y)



if __name__ == '__main__':

    dg = DataGenerator(['../clean_signals/chb01'], do_shuffle = True, n_processes = 16, do_preemphasis = True)

    print("loaded %d signal vectors" % dg.num_signal_vectors)
    print("loaded %d data samples" % dg.num_samples)
    print("available %d batches" % len(dg))

    for x, y in dg:
        print(x.shape, y.shape)
