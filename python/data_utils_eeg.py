import os
import sys
import bz2
import pickle
import _pickle as cPickle
from tqdm import tqdm
import numpy
from sklearn.preprocessing import PolynomialFeatures

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
    data_fbank = decompress_pickle(filename + '.fbank.pbz2')
    data_td_stats = decompress_pickle(filename + '.td_stats.pbz2')
    #
    return data_fbank, data_td_stats, labels, timestamps


class DataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, index_filenames = None,
                        batch_size = 20,
                        do_shuffle = False,
                        do_standard_scaling = True,
                        in_training_mode = False,
                        balance_classes = True,
                        verbose = 0):
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

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale features to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.
            
            :param boolean balance_classes:
                Flag to indicate whether to balance the classes of each batch of samples.

            :param int verbose:
                Level of verbosity.

        '''
        #
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
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
        self.whole_data = list()
        self.data = list()
        self.data_class_1 = list()
        self.input_shape = None
        for fname in self.filenames:
            ignore_this_file = False
            if fname[0] != '#':
                x_fbank, x_stats, labels, timestamp = load_file(fname, verbose = verbose)
                #print(x_fbank.shape, x_stats.shape, labels.shape, timestamp.shape)
                #(1799,21,8), (1799,21,6), (1799,), (1799,)
                x = numpy.concatenate((x_fbank, x_stats), axis=2)
                self.whole_data.append(x)

                if self.input_shape is None:
                    self.input_shape = x.shape[1:]
                elif self.input_shape != x.shape[1:]:
                    raise Exception('unexpected input shape: ' + str(x.shape))
            #
            else:
                ignore_this_file = True

            if not ignore_this_file:
                self.num_samples += len(x)
                for i in range(len(x)):
                    if self.balance_classes:
                        if   labels[i] == 0:         self.data.append([x, labels[i], timestamp[i], i])
                        elif labels[i] == 1: self.data_class_1.append([x, labels[i], timestamp[i], i])
                    else:
                        self.data.append([x, labels[i], timestamp[i], i])
                #
            #
        #

        if do_standard_scaling:
            if verbose > 0: print('Scaling data...')
            if in_training_mode:
                # Filter bank features
                fbank_means = []
                fbank_stddevs = []
                #
                # Time Domain Statistics
                mean_means = []
                mean_stddevs = []
                std_means = []
                std_stddevs = []
                kurt_means = []
                kurt_stddevs = []
                skew_means = []
                skew_stddevs = []
                #
                # Hjorth Parameters: Mobility, Complexity
                mob_means = []
                mob_stddevs = []
                comp_means = []
                comp_stddevs = []
                #
                counts = []     # Data counter
                

                for x in tqdm(self.whole_data):
                    x_fbank = x[:,:,:8]
                    x_mean = x[:,:,8]
                    x_std = x[:,:,9]
                    x_kurt = x[:,:,10]
                    x_skew = x[:,:,11]
                    x_mob = x[:,:,12]
                    x_comp = x[:,:,13]
                    
                    fbank_means.append(x_fbank.mean())
                    fbank_stddevs.append(x_fbank.std())
                    mean_means.append(x_mean.mean(axis=0))
                    mean_stddevs.append(x_mean.std(axis=0))
                    std_means.append(x_std.mean(axis=0))
                    std_stddevs.append(x_std.std(axis=0))
                    kurt_means.append(x_kurt.mean(axis=0))
                    kurt_stddevs.append(x_kurt.std(axis=0))
                    skew_means.append(x_skew.mean(axis=0))
                    skew_stddevs.append(x_skew.std(axis=0))
                    mob_means.append(x_mob.mean(axis=0))
                    mob_stddevs.append(x_mob.std(axis=0))
                    comp_means.append(x_comp.mean(axis=0))
                    comp_stddevs.append(x_comp.std(axis=0))
                    counts.append(len(x))
                #
                self.fbank_mean = sum([m * c for m, c in zip(fbank_means, counts)])
                self.fbank_std = sum([s * c for s, c in zip(fbank_stddevs, counts)])
                self.fbank_mean /= 8 * sum(counts) # We use one mean and std for the 8 channels in frequency
                self.fbank_std /= 8 * sum(counts)
                
                self.td_stats_means = []    # Lists with time domain statistics means and stdevs
                self.td_stats_stddevs = []  # 0: mean, 1: std, 2: kurtosis, 3: skewness 4: Mobility 5: Complexity

                for ft_means, ft_stddevs in [(mean_means, mean_stddevs), (std_means, std_stddevs),
                                        (kurt_means, kurt_stddevs), (skew_means, skew_stddevs),
                                        (mob_means, mob_stddevs), (comp_means, comp_stddevs)]:

                    mean = sum([m * c for m, c in zip(ft_means, counts)])
                    std = sum([m * c for m, c in zip(ft_stddevs, counts)])
                    mean /= sum(counts)
                    std /= sum(counts)
                    self.td_stats_means.append(mean)
                    self.td_stats_stddevs.append(std)

                print("STATS")
                print("fbank_mean values", self.fbank_mean)
                print("fbank_std values", self.fbank_std)
                print("td_stats_mean mean values", self.td_stats_means[0])
                print("td_stats_mean std values", self.td_stats_stddevs[0])
                print("td_stats_std mean values", self.td_stats_means[1])
                print("td_stats_std std values", self.td_stats_stddevs[1])
                print("td_stats_kurtosis mean values", self.td_stats_means[2])
                print("td_stats_kurtosis std values", self.td_stats_stddevs[2])
                print("td_stats_skew mean values", self.td_stats_means[3])
                print("td_stats_skew std values", self.td_stats_stddevs[3])
                print("td_stats_mob mean values", self.td_stats_means[4])
                print("td_stats_mob std values", self.td_stats_stddevs[4])
                print("td_stats_compl mean values", self.td_stats_means[5])
                print("td_stats_compl std values", self.td_stats_stddevs[5])
                print('\n\n MAX-MIN')

                print("fbank_mean values(max, min): ", numpy.amax(self.fbank_mean), numpy.amin(self.fbank_mean))
                print("fbank_mean values(max, min): ", numpy.amax(self.fbank_std), numpy.amin(self.fbank_std))
                print("td_stats_mean mean values(max, min): ", numpy.amax(self.td_stats_means[0]), numpy.amin(self.td_stats_means[0]))
                print("td_stats_mean std values(max, min): ", numpy.amax(self.td_stats_stddevs[0]), numpy.amin(self.td_stats_stddevs[0]))
                print("td_stats_std mean values(max, min): ", numpy.amax(self.td_stats_means[1]), numpy.amin(self.td_stats_means[1]))
                print("td_stats_std std values(max, min): ", numpy.amax(self.td_stats_stddevs[1]), numpy.amin(self.td_stats_stddevs[1]))
                print("td_stats_kurtosis mean values(max, min): ", numpy.amax(self.td_stats_means[2]), numpy.amin(self.td_stats_means[2]))
                print("td_stats_kurtosis std values(max, min): ", numpy.amax(self.td_stats_stddevs[2]), numpy.amin(self.td_stats_stddevs[2]))
                print("td_stats_skew mean values(max, min): ", numpy.amax(self.td_stats_means[3]), numpy.amin(self.td_stats_means[3]))
                print("td_stats_skew std values(max, min): ", numpy.amax(self.td_stats_stddevs[3]), numpy.amin(self.td_stats_stddevs[3]))
                print("td_stats_mob mean values(max, min): ", numpy.amax(self.td_stats_means[4]), numpy.amin(self.td_stats_means[4]))
                print("td_stats_mob std values(max, min): ", numpy.amax(self.td_stats_stddevs[4]), numpy.amin(self.td_stats_stddevs[4]))
                print("td_stats_compl mean values(max, min): ", numpy.amax(self.td_stats_means[5]), numpy.amin(self.td_stats_means[5]))
                print("td_stats_compl std values(max, min): ", numpy.amax(self.td_stats_stddevs[5]), numpy.amin(self.td_stats_stddevs[5]))
                
                stats_dict = {}
                stats_dict['fbank_mean'] = self.fbank_mean
                stats_dict['fbank_std'] = self.fbank_mean
                stats_dict['td_stats_means'] = self.td_stats_means
                stats_dict['td_stats_stddevs'] = self.td_stats_stddevs
                with open('models/eeg_statistics.pkl', 'wb') as f:
                    pickle.dump(stats_dict, f)
                    f.close()
            #
            else:
                with open('models/eeg_statistics.pkl', 'rb') as f:
                    stats_dict = pickle.load(f)
                    f.close()
                    self.fbank_mean = stats_dict['fbank_mean']
                    self.fbank_std = stats_dict['fbank_std']
                    self.td_stats_means = stats_dict['td_stats_means']
                    self.td_stats_stddevs = stats_dict['td_stats_stddevs']
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
        poly = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = False)
        #
        if self.balance_classes:
            B = self.batch_size // 2
            for b in range(B):
                pos = (B * batch_index + b) % len(self.indexes)

                x, y, t, i = self.data[self.indexes[pos]]
                x = x[i]    # x was the reference to all the data from a full file

                if self.do_standard_scaling:
                    x = self.scale_data(x)

                x = poly.fit_transform(x.T).T

                X.append(x)
                Y.append(y)
                T.append(t)

                x, y, t, i = self.data_class_1[self.indexes_class_1[pos]]
                x = x[i]    # x was the reference to all the data from a full file

                if self.do_standard_scaling:
                    x = self.scale_data(x)

                x = poly.fit_transform(x.T).T

                X.append(x)
                Y.append(y)
                T.append(t)
        else:
            B = self.batch_size
            for b in range(B):
                pos = (B * batch_index + b) % len(self.indexes)

                x, y, t, i = self.data[self.indexes[pos]]
                x = x[i]    # x was the reference to all the data from a whole file

                if self.do_standard_scaling:
                    x = self.scale_data(x)

                x = poly.fit_transform(x.T).T

                X.append(x)
                Y.append(y)
                T.append(t)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)
        T = numpy.array(T)

        return (X, Y, T)


    def scale_data(self, X):
        '''
        Scales the data to mean zero and standard deviation 1.

        :param self  Reference to the current object.

        :param numpy.array  X   An array with the data.

        :return numpy.array X   An array with the scaled data.
        '''

        X[:,:8] = (X[:,:8] - self.fbank_mean) / self.fbank_std
        X[:,8] = (X[:,8] - self.td_stats_means[0]) / self.td_stats_stddevs[0]
        X[:,9] = (X[:,9] - self.td_stats_means[1]) / self.td_stats_stddevs[1]
        X[:,10] = (X[:,10] - self.td_stats_means[2]) / self.td_stats_stddevs[2]
        X[:,11] = (X[:,11] - self.td_stats_means[3]) / self.td_stats_stddevs[3]
        X[:,12] = (X[:,12] - self.td_stats_means[4]) / self.td_stats_stddevs[4]
        X[:,13] = (X[:,13] - self.td_stats_means[5]) / self.td_stats_stddevs[5]
        #
        return X


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

    dg = DataGenerator(index_filenames, in_training_mode=True, balance_classes = True, verbose = 2)

    print("loaded %d data samples" % dg.num_samples)
    print("available %d batches" % len(dg))

    counter = 0
    samples = 0

    for i in range(len(dg)):
        x, y, t = dg[i]
        #print(x.shape, y.shape, t.shape)
        counter += 1
        samples += len(x)
        print('%20d' % counter, '%20d' % samples, end = '\r')
    print(f'loaded a total of {samples} samples grouped in {counter} batches of {dg.batch_size} samples')
