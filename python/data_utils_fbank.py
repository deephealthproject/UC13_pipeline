import os
import sys
import bz2
import pickle
import _pickle as cPickle
import numpy
import datetime

from data_utils import decompress_pickle

from random import shuffle

from multiprocessing import Pool, TimeoutError


def load_file(filename, verbose = 0):
    '''
    Loads the contents of one file.

    :param str filename:
        Name of the file from which to load data.

    :param int verbose:
        Level of verbosity.

    :return: A tuple with the sequences of vectors, labels and timestamps.
    '''
    #
    if verbose > 0:
        print('loading', filename)
    #
    labels     = numpy.load(filename + '.labels.npy', allow_pickle = True)
    timestamps = numpy.load(filename + '.timestamp.npy', allow_pickle = True)
    data       = decompress_pickle(filename + '.fbank.pbz2')
    #
    return data, labels, timestamps


class DataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self,  index_filenames = None,
                        batch_size = 20,
                        n_processes = 4,
                        do_shuffle = False,
                        balance_classes = True):
        '''

            Constructor to create objects of the class **DataGenerator** and loads all the data.
            Future implementations will pave the way to load data from files on-the-fly in order to allow work
            with large enough datasets.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int batch_size:
                Number of samples per batch.

            :param int n_processes:
                Number of threads to use for loading data from files.

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.
        '''
        #
        self.mean = 2.4653737460652594
        self.std = 0.7361729493017065
        #
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.balance_classes = balance_classes
        #
        self.num_batches = 0
        self.num_samples = 0
        self.num_signal_vectors = 0
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    self.filenames.append(l.strip())
                f.close()
        #
        self.data = list()
        self.data_class_1 = list()
        self.input_shape = None
        for fname in self.filenames:
            ignore_this_file = False
            if fname[0] != '#':
                x, y, t = load_file(fname, verbose = 2)
                x = (x - self.mean) / self.std
                #print(x.shape, y.shape, t.shape)
                if self.input_shape is None:
                    self.input_shape = x.shape[1:]
                elif self.input_shape != x.shape[1:]:
                    if x.shape[1] == 24 and self.input_shape[0] == 23:
                        x = x[:,0:23,:]
                    else:
                        #raise Exception('unexpected input shape: ' + str(x.shape))
                        print('unexpected input shape: ' + str(x.shape[1:]))
                        ignore_this_file = True
            else:
                ignore_this_file = True

            if not ignore_this_file:
                self.num_samples += len(x)
                for i in range(len(x)):
                    k = min(len(y) - 1, i) # because some label and timestamp files contain less samples data
                    if k < i:
                        _t_ = t[k] + (i - k) * datetime.timedelta(seconds = 5)
                    else:
                        _t_ = t[k]

                    if self.balance_classes:
                        if   y[k] == 0:         self.data.append([x, y[k], _t_, i]) # warning
                        elif y[k] == 1: self.data_class_1.append([x, y[k], _t_, i])
                    else:
                        self.data.append([x, y[k], _t_, i])
                #
            #
        #
        self.num_samples = len(self.data) + len(self.data_class_1)
        if self.balance_classes and len(self.data_class_1) > 0:
            self.indexes_class_1 = [(i % len(self.data_class_1)) for i in range(len(self.data))]
        else:
            self.indexes_class_1 = None
            self.balance_classes = False
        self.indexes = [i for i in range(len(self.data))]
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
            shuffle(self.indexes)
            if self.indexes_class_1 is not None:
                shuffle(self.indexes_class_1)
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
        T = list()
        #
        if self.balance_classes:
            B = self.batch_size // 2
            for b in range(B):
                pos = (B * batch_index + b) % len(self.indexes)

                x, y, t, i = self.data[self.indexes[pos]]

                X.append(numpy.expand_dims(x[i], axis = 0))
                Y.append(y)
                T.append(t)

                x, y, t, i = self.data_class_1[self.indexes_class_1[pos]]

                X.append(numpy.expand_dims(x[i], axis = 0))
                Y.append(y)
                T.append(t)
        else:
            B = self.batch_size
            for b in range(B):
                pos = (B * batch_index + b) % len(self.indexes)

                x, y, t, i = self.data[self.indexes[pos]]

                X.append(numpy.expand_dims(x[i], axis = 0))
                Y.append(y)
                T.append(t)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)
        T = numpy.array(T)
        #
        return (X, Y, T)



if __name__ == '__main__':
    #
    index_filenames = list()
    #
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])
    #
    if index_filenames is None  or  len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')

    dg = DataGenerator(index_filenames, n_processes = 16)

    print("loaded %d data samples" % dg.num_samples)
    print("available %d batches" % len(dg))

    counter = 0
    samples = 0
    #for x, y, t in dg:
    for i in range(len(dg)):
        x, y, t = dg[i]
        #print(x.shape, y.shape, t.shape)
        counter += 1
        samples += len(x)
        print('%20d' % counter, '%20d' % samples, end = '\r')
    print(f'loaded a total of {samples} samples grouped in {counter} batches of {dg.batch_size} samples')
