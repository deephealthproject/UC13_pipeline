import os
import sys
import datetime
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

class SimpleDataGenerator:
    '''
        Class for fetching data and provide single samples..
    '''

    def __init__(self, index_filenames = None, verbose = 0):
        '''

            Constructor to create objects of the class **SimpleDataGenerator** that will load data as required.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int verbose:
                Level of verbosity.

        '''
        #
        self.verbose = verbose
        self.filenames = list()
        for index_fname in index_filenames:
            with open(index_fname, 'r') as f:
                for line in f:
                    l = line.strip()
                    if l[0] != '#': 
                        self.filenames.append(l)
                f.close()
        #
        self.index_current_file = -1
        self.index_current_sample = -1
        self.fetch_next_file()

    def __len__(self):
        raise Exception('This objects cannot compute the total number of samples it will provide')

    def fetch_next_file(self):
        self.index_current_file += 1
        if self.index_current_file < len(self.filenames):
            data_fbank, data_td_stats, labels, timestamps = load_file(self.filenames[self.index_current_file], verbose = self.verbose)
            self.X = numpy.concatenate((data_fbank, data_td_stats), axis=2)
            #self.X = self.X.reshape(-1, self.X.shape[2])
            self.labels = labels
            self.timestamps = timestamps
            if self.verbose > 0:
                print(self.filenames[self.index_current_file], data_fbank.shape, data_td_stats.shape, self.X.shape, len(self.labels), len(self.timestamps))
        else:
            self.X = None
        self.index_current_sample = -1
        

    def next(self):
        self.index_current_sample += 1
        if self.X is None  or  self.index_current_sample >= len(self.X):
            self.fetch_next_file()
            if self.X is None:
                return None, None

        return self.X[self.index_current_sample], self.labels[self.index_current_sample]

    def next_block(self, n=10):
        X = list()
        L = list()
        for i in range(n):
            x, l = self.next()
            if x is not None:
                X.append(x)
                L.append(l)
        if len(X) > 0:
            return X, L
        else:
            return None, None
    # --------------------------------------------------------------------------------------------


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
            This generator provides batches applying polynomical expansion to the signal channels.
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
                self.fbank_mean /= sum(counts) # We use one mean and std for the 8 channels in frequency
                self.fbank_std /= sum(counts)
                
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

        print("Data Generator created!")
        print("Shape of the data: ", (self.batch_size,) + self.input_shape)

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
        if (self.num_samples % self.batch_size) != 0:
            self.num_batches += 1

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
                x = numpy.array(x[i], dtype=numpy.float64)    # x was the reference to all the data from a whole file

                if self.do_standard_scaling:
                    x = self.scale_data(x)
                
                x = poly.fit_transform(x.T).T

                X.append(x)
                Y.append(y)
                T.append(t)

                x, y, t, i = self.data_class_1[self.indexes_class_1[pos]]   
                x = numpy.array(x[i], dtype=numpy.float64)    # x was the reference to all the data from a whole file

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
                x = numpy.array(x[i], dtype=numpy.float64) # x was the reference to all the data from a whole file

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


class SequenceDataGenerator:
    '''
        Class for preloading data and provide sequence batches.
    '''

    def __init__(self, index_filenames = None,
                        batch_size = 20,
                        sequence_length = 50,
                        subsampling_period = 2, # in seconds
                        max_interictal = 3600 * 2, # in seconds
                        min_interictal = 3600, # in seconds
                        do_shuffle = False,
                        do_standard_scaling = True,
                        in_training_mode = False,
                        verbose = 0):
        '''

            Constructor to create objects of the class **SequenceDataGenerator** and loads all the data.
            This generator provides sequences of data, using only one signal channel for each sample in the sequence.
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

            :param sequence_length:
                Length of the sequences of data.
            
            :param subsampling_period:
                Subsampling period that was used when the data was sampled.

            :param max_interictal:
                Maximum period allowed for interictal stages. If an interictal stage is longer than max_interictal,
                it is cropped to the maximum length allowed.

            :param min_interictal:
                Minimum period allowed for interictal stages. If an interictal stage is shorter than this value
                the following seizure is not processed.

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale features to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

            :param int verbose:
                Level of verbosity.

        '''
        #
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
        #
        self.subsampling_period = subsampling_period
        self.max_interictal = max_interictal
        self.min_interictal = min_interictal
        #
        self.num_batches = 0
        self.num_signal_vectors = 0
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    self.filenames.append(l.strip())
                f.close()
        #

        self.files = list()
        self.indexes = list()
        self.input_shape = None

        previous_file = None
        current_interictal_length = 0
        for fname in self.filenames:
            ignore_this_file = False
            if fname[0] != '#':
                x_fbank, x_stats, labels, timestamp = load_file(fname, verbose = verbose)
                #(1799,21,8), (1799,21,6), (1799,), (1799,)
                x = numpy.concatenate((x_fbank, x_stats), axis=2)
                done = False

                if previous_file is not None:
                    if previous_file[3]:
                        # The previous file ends with interictal
                        # Calculate first pair and take interictal from previous file if needed
                        interictal_indices = list()
                        # Labels: preictal = 2, interictal = 0
                        for j in range(len(x)):
                            if labels[j] == 0:
                                interictal_indices.append(j)
                            if labels[j] == 2:
                                break
                        i = j
                        if i == len(x) - 1 and labels[i] != 2:
                            # We arrived to the end of the file
                            # Check timestamp
                            if timestamp[0] - previous_file[2][-1] < datetime.timedelta(seconds = 1000000): # Remember to change this number in the future!!!
                                new_x = numpy.concatenate((previous_file[0], x[interictal_indices]), axis = 0)
                            else:
                                new_x = x[interictal_indices]
                            #
                            new_labels = numpy.zeros((len(new_x))) # Only zeros because all labels are interictal
                            previous_file = (new_x, new_labels, timestamp, True, False)
                            print("end of file (out of loop)")
                            done = True

                        else:
                            # We arrived to a interictal-preictal stage
                            for j in range(i, len(x)):
                                if labels[j] == 2:
                                    preictal_indices.append(j)
                                else:
                                    break
                            #
                            i = j
                            pair = interictal_indices + preictal_indices
                            new_x = numpy.concatenate((previous_file[0], x[pair]), axis = 0)
                            new_labels = numpy.zeros(len(pair))
                            new_labels[len(interictal_indices):] = 1
                            new_labels = numpy.concatenate((previous_file[1], new_labels), axis = 0)
                            assert len(new_x) == len(new_labels)
                            #
                            if i == len(x) - 1:
                                # The file ends with preictal data
                                previous_file = (new_x, new_labels, timestamp, False, True)
                                print("end of file with preictal (out of loop)")
                                done = True

                            else:
                                # Save the interictal-preictal pair
                                for k in range(len(new_x)):
                                    self.whole_data.append((new_x[k], new_labels[k]))
                #

                while (not done):
                    interictal_indices = list()
                    preictal_indices = list()
                    # Labels: preictal = 2, interictal = 0
                    for j in range(i, len(x)):
                        if labels[j] == 0:
                            interictal_indices.append(j)
                        if labels[j] == 2:
                            break
                    #
                    i = j
                    current_interictal_length += len(interictal_indices) * self.subsampling_period
                    if (i == len(x) - 1) and (labels[i] != 2):
                        # We arrived to the end of the file
                        # previous_file: data, labels, timestamp of prev file, ends with interictal?, ends with preictal?
                        previous_file = (x[interictal_indices], labels[interictal_indices], timestamp[interictal_indices], True) 
                        print("end of file with interictal")
                        done = True

                    else:
                        # We arrived to a interictal-preictal stage
                        
                        if (current_interictal_length > self.max_interictal):
                            # If the interictal length is greater than maximum, crop the interictal stage
                            interictal_indices = interictal_indices[-(self.max_interictal // self.subsampling_period):]

                        elif (current_interictal_length < self.min_interictal):
                            # Ignore the following seizure, not enough interictal data - NOT IMPLEMENTED YET
                            #interictal_indices = list()
                            #ignore_seizure = True
                            pass
                        #

                        findex = len(self.files)
                        self.files.append(x[interictal_indices])
                        # Generate indexes of the current file
                        indexes = numpy.arange(self.sequence_length - 1, len(x[interictal_indices]), 1)
                        for t in indexes:
                            self.indexes.append((findex, t, labels[interictal_indices][t], timestamp[interictal_indices][t]))

                        
                        for j in range(i, len(x)):
                            if labels[j] == 0:
                                new_interictal = True
                                break
                        #

                        findex = len(self.files)
                        self.files.append(x[i:j+1])
                        # Generate indexes of the current file
                        indexes = numpy.arange(self.sequence_length - 1, len(x[i:j+1]), 1)
                        for t in indexes:
                            self.indexes.append((findex, t, labels[i:j+1][t], timestamp[i:j+1][t]))

                        done = True

                        if new_interictal:
                            # New interictal data found, we ignore it. Future implementations will process it
                    #
                #

                findex = len(self.files)
                self.files.append(x)

                # Generate indexes of the current file
                indexes = numpy.arange(self.sequence_length - 1, len(x), 1)
                
                #for c in range(x.shape[1]):
                #    for t in indexes:
                #        self.indexes.append((findex, t, c, labels[t], timestamp[t]))
                for t in indexes:
                        self.indexes.append((findex, t, labels[t], timestamp[t]))
                #

                if self.input_shape is None:
                    self.input_shape = x.shape[1:]
                elif self.input_shape != x.shape[1:]:
                    raise Exception('unexpected input shape: ' + str(x.shape))
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
                

                for x in tqdm(self.files):

                    x_fbank = x[:, :, :8]
                    x_mean = x[:, :, 8]
                    x_std = x[:, :, 9]
                    x_kurt = x[:, :, 10]
                    x_skew = x[:, :, 11]
                    x_mob = x[:, :, 12]
                    x_comp = x[:, :, 13]
                    
                    fbank_means.append(x_fbank.mean())
                    fbank_stddevs.append(x_fbank.std())
                    mean_means.append(x_mean.mean())
                    mean_stddevs.append(x_mean.std())
                    std_means.append(x_std.mean())
                    std_stddevs.append(x_std.std())
                    kurt_means.append(x_kurt.mean())
                    kurt_stddevs.append(x_kurt.std())
                    skew_means.append(x_skew.mean())
                    skew_stddevs.append(x_skew.std())
                    mob_means.append(x_mob.mean())
                    mob_stddevs.append(x_mob.std())
                    comp_means.append(x_comp.mean())
                    comp_stddevs.append(x_comp.std())
                    counts.append(x.shape[0] * x.shape[1])
                #

                self.fbank_mean = sum([m * c for m, c in zip(fbank_means, counts)])
                self.fbank_std = sum([s * c for s, c in zip(fbank_stddevs, counts)])
                self.fbank_mean /= sum(counts) # We use one mean and std for the 8 channels in frequency
                self.fbank_std /= sum(counts)
                
                self.td_stats_means = []    # Lists with time domain statistics means and stdevs
                self.td_stats_stddevs = []  # 0: mean, 1: std, 2: kurtosis, 3: skewness 4: Mobility 5: Complexity

                for ft_means, ft_stddevs in [(mean_means, mean_stddevs), (std_means, std_stddevs),
                                        (kurt_means, kurt_stddevs), (skew_means, skew_stddevs),
                                        (mob_means, mob_stddevs), (comp_means, comp_stddevs)]:

                    mean = sum([m * c for m, c in zip(ft_means, counts)])
                    std = sum([s * c for s, c in zip(ft_stddevs, counts)])
                    mean /= sum(counts)
                    std /= sum(counts)
                    self.td_stats_means.append(mean)
                    self.td_stats_stddevs.append(std)
                #
                
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
            
            #print(self.fbank_mean)
            #print(self.fbank_std)
            #for m, s in zip(self.td_stats_means, self.td_stats_stddevs):
            #    print(m)
            #    print(s)
        #

        print("Data Generator created!")
        print("Shape of the data: ", (self.batch_size,) + (self.sequence_length,) + self.input_shape)

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

        self.num_batches = len(self.indexes) // self.batch_size
        if (len(self.indexes) % self.batch_size) != 0:
            self.num_batches += 1

        if self.do_shuffle:
            shuffle(self.indexes)


    def __getitem__(self, batch_index : int):
        '''
        Returns the batch of samples specified by the index.

        :param self: Reference to the current object.

        :param int batch_index: Index of the batch to retrieve.

        :return: A tuple with three objects, a batch of samples, the corresponding labels and the timestamps.
        '''

        X = list()
        Y = list()
        T = list()
        
        B = self.batch_size
        for b in range(B):
            pos = (B * batch_index + b) % len(self.indexes)

            #findex, t, c, label, timestamp = self.indexes[pos]
            #x = self.files[findex][t - self.sequence_length + 1 : t + 1, c].copy()
            findex, t, label, timestamp = self.indexes[pos]
            x = self.files[findex][t - self.sequence_length + 1 : t + 1, :].copy()
            #

            if self.do_standard_scaling:
                x = self.scale_data(x)
            
            X.append(x.reshape((self.sequence_length, 21 * 14)))
            Y.append(label) # The label of the sequence will be the label of the last sample
            T.append(timestamp) # The timestamp of the sequence will be the one of the last sample
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
        #for x in X:
        #    print(' '.join('{:g}'.format(a) for a in x))

        X[:,:8] = (X[:,:8] - self.fbank_mean) / self.fbank_std
        X[:,8] = (X[:,8] - self.td_stats_means[0]) / self.td_stats_stddevs[0]
        X[:,9] = (X[:,9] - self.td_stats_means[1]) / self.td_stats_stddevs[1]
        X[:,10] = (X[:,10] - self.td_stats_means[2]) / self.td_stats_stddevs[2]
        X[:,11] = (X[:,11] - self.td_stats_means[3]) / self.td_stats_stddevs[3]
        X[:,12] = (X[:,12] - self.td_stats_means[4]) / self.td_stats_stddevs[4]
        X[:,13] = (X[:,13] - self.td_stats_means[5]) / self.td_stats_stddevs[5]
        
        #print(X.shape)
        #for x in X:
        #    print(' '.join('{:g}'.format(a) for a in x))
        #input()
        #

        return X


class PairDataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, index_filenames = None,
                        batch_size = 20,
                        subsampling_period = 2, # in seconds
                        do_shuffle = False,
                        do_standard_scaling = True,
                        in_training_mode = False,
                        verbose = 0):
        '''

            Constructor to create objects of the class **PairDataGenerator** and loads all the data.
            This generator provides batches of samples from interictal-preictal periods in the signal.
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

            :param int subsampling_period:
                Subsampling period (or shift) used when the data was sampled.

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale features to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

            :param int verbose:
                Level of verbosity.

        '''
        #
        self.batch_size = batch_size
        self.subsampling_period = subsampling_period
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
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
        self.input_shape = None
        previous_file = None
        previous_interictal = None
        for fname in self.filenames:

            if fname[0] != '#':
                x_fbank, x_stats, labels, timestamp = load_file(fname, verbose = verbose)
                #print(x_fbank.shape, x_stats.shape, labels.shape, timestamp.shape)
                #(1799,21,8), (1799,21,6), (1799,), (1799,)
                x = numpy.concatenate((x_fbank, x_stats), axis=2, dtype=numpy.float64)

                i = 0
                done = False
                if previous_file is not None:
                    if previous_file[4]:
                        # The previous file ends with preictal
                        new_file_preictal = list()
                        for j in range(len(x)):
                            if labels[j] == 2: # 2 is label for preictal
                                new_file_preictal.append(j)
                            else:
                                break
                        #
                        i = j
                        new_x = numpy.concatenate((previous_file[0], new_x[new_file_preictal]), axis = 0)
                        new_labels = numpy.ones((len(new_file_preictal)))
                        new_labels = numpy.concatenate((previous_file[1], new_labels), axis = 0)
                        assert len(new_x) == len(new_labels)
                        #
                        if i == len(x) - 1:
                            # Current file also ends with preictal
                            previous_file = (new_x, new_labels, timestamp[new_file_preictal], False, True)
                            print("end of file with preictal 2 (out of loop)")
                            done = True
                        else:
                            # Save the interictal-preictal pair
                            for k in range(len(new_x)):
                                self.whole_data.append((new_x[k], new_labels[k]))
                    #
                    else:
                        # The previous file does not end with preictal
                        if previous_file[3]:
                            # The previous file ends with interictal
                            # Calculate first pair and take interictal from previous file if needed
                            interictal_indices = list()
                            preictal_indices = list()
                            # Labels: preictal = 2, interictal = 0
                            for j in range(len(x)):
                                if labels[j] == 0:
                                    interictal_indices.append(j)
                                if labels[j] == 2:
                                    break
                            i = j
                            if i == len(x) - 1 and labels[i] != 2:
                                # We arrived to the end of the file
                                # Check timestamp
                                if timestamp[0] - previous_file[2][-1] < datetime.timedelta(seconds = 1000000): # Remember to change this number in the future!!!
                                    new_x = numpy.concatenate((previous_file[0], x[interictal_indices]), axis = 0)
                                else:
                                    new_x = x[interictal_indices]
                                #
                                new_labels = numpy.zeros((len(new_x))) # Only zeros because all labels are interictal
                                previous_file = (new_x, new_labels, timestamp, True, False)
                                print("end of file (out of loop)")
                                done = True

                            else:
                                # We arrived to a interictal-preictal stage
                                for j in range(i, len(x)):
                                    if labels[j] == 2:
                                        preictal_indices.append(j)
                                    else:
                                        break
                                #
                                i = j
                                pair = interictal_indices + preictal_indices
                                new_x = numpy.concatenate((previous_file[0], x[pair]), axis = 0)
                                new_labels = numpy.zeros(len(pair))
                                new_labels[len(interictal_indices):] = 1
                                new_labels = numpy.concatenate((previous_file[1], new_labels), axis = 0)
                                assert len(new_x) == len(new_labels)
                                #
                                if i == len(x) - 1:
                                    # The file ends with preictal data
                                    previous_file = (new_x, new_labels, timestamp, False, True)
                                    print("end of file with preictal (out of loop)")
                                    done = True

                                else:
                                    # Save the interictal-preictal pair
                                    for k in range(len(new_x)):
                                        self.whole_data.append((new_x[k], new_labels[k]))
                #

                while (not done):
                    interictal_indices = list()
                    preictal_indices = list()
                    # Labels: preictal = 2, interictal = 0
                    for j in range(i, len(x)):
                        if labels[j] == 0:
                            interictal_indices.append(j)
                        if labels[j] == 2:
                            break
                    #
                    i = j
                    if (i == len(x) - 1) and (labels[i] != 2):
                        # We arrived to the end of the file
                        # previous_file: data, labels, timestamp of prev file, ends with interictal?, ends with preictal?
                        previous_file = (x[interictal_indices], labels[interictal_indices], timestamp, True, False) 
                        print("end of file with interictal")
                        done = True

                    else:
                        # We arrived to a interictal-preictal stage
                        for j in range(i, len(x)):
                            if labels[j] == 2:
                                preictal_indices.append(j)
                            else:
                                break
                        #
                        i = j
                        if i == len(x) - 1:
                            # The file ends with preictal data
                            if len(interictal_indices) == 0:
                                previous_file = None  # Ignore file because there is not interictal data
                            else:
                                pair = interictal_indices + preictal_indices
                                previous_file = (x[pair], labels[pair], timestamp, False, True)
                            print("end of file with preictal (out of loop)")
                            done = True

                        else:
                            # Save the interictal-preictal pair
                            pair = interictal_indices + preictal_indices
                            new_x = x[pair]
                            new_labels = numpy.zeros((len(pair)))
                            new_labels[len(interictal_indices):] = 1 # Save label 0 for interictal, 1 for preictal
                            for k in range(len(new_x)):
                                self.whole_data.append((new_x[k], new_labels[k]))
                    #
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
                

                for x, _ in tqdm(self.whole_data):
                    x_fbank = x[:,:8]
                    x_mean = x[:,8]
                    x_std = x[:,9]
                    x_kurt = x[:,10]
                    x_skew = x[:,11]
                    x_mob = x[:,12]
                    x_comp = x[:,13]
                    
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
                #
                self.fbank_mean = sum(fbank_means) / len(self.whole_data)
                self.fbank_std = sum(fbank_stddevs) / len(self.whole_data)
                
                self.td_stats_means = []    # Lists with time domain statistics means and stdevs
                self.td_stats_stddevs = []  # 0: mean, 1: std, 2: kurtosis, 3: skewness 4: Mobility 5: Complexity

                for ft_means, ft_stddevs in [(mean_means, mean_stddevs), (std_means, std_stddevs),
                                        (kurt_means, kurt_stddevs), (skew_means, skew_stddevs),
                                        (mob_means, mob_stddevs), (comp_means, comp_stddevs)]:

                    mean = sum(ft_means) / len(self.whole_data)
                    std = sum(ft_stddevs) / len(self.whole_data)
                    self.td_stats_means.append(mean)
                    self.td_stats_stddevs.append(std)
                
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
        self.num_samples = len(self.whole_data)
        self.indexes = [i for i in range(len(self.whole_data))]
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
        self.num_batches = len(self.indexes) // self.batch_size
        if (len(self.indexes) % self.batch_size) != 0:
            self.num_batches += 1

        if self.do_shuffle:
            shuffle(self.indexes)
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
        poly = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = False)
        #
        B = self.batch_size
        for b in range(B):
            pos = (B * batch_index + b) % len(self.indexes)

            x, y = self.whole_data[self.indexes[pos]]
            x = numpy.array(x, dtype=numpy.float64)

            if self.do_standard_scaling:
                x = self.scale_data(x)

            x = poly.fit_transform(x.T).T

            X.append(x)
            Y.append(y)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)

        return (X, Y)


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

    #dg = DataGenerator(index_filenames, in_training_mode = True, balance_classes = True, verbose = 2)
    dg = SequenceDataGenerator(index_filenames, in_training_mode=True, verbose = 2)
    
    print("available %d batches" % len(dg))

    counter = 0
    samples = 0

    print(len(dg))
    for i in range(len(dg)):
        x, y, t = dg[i]
        #print(x.shape, y.shape) #, t.shape)
        counter += 1
        samples += len(x)
        print('%20d' % counter, '%20d' % samples, end = '\r')
    print(f'loaded a total of {samples} samples grouped in {counter} batches of {dg.batch_size} samples')
