import os
import sys
import bz2
import pickle
import _pickle as cPickle
import numpy
from tqdm import tqdm

from random import shuffle
import Preprocessor as preprocessor
from multiprocessing import Pool, TimeoutError, cpu_count


def decompress_pickle(filename):
    '''
    Load any compressed pickle file

    :param str filename:
        Name of the file from which to load data.

    :return: The contents of the file.
    '''
    f = bz2.BZ2File(filename, 'rb')
    data = None
    try:
        data = cPickle.load(f)
    except:
        sys.stderr.write(f'error loading data from {filename}\n')
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
    if signal_dict is None:
        return None
    metadata = signal_dict.get('metadata')
    if verbose > 1:
        print(signal_dict.keys())
        print(metadata.keys())

    signal_ids = metadata['channels']
    try:
        num_seizures = metadata['seizures']
    except:
        num_seizures = 0
    try:
        episodes = metadata['times']
    except:
        episodes = None

    if episodes is not None and len(episodes) > 0:
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
    if separate_seizures and episodes is not None:
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
        try:
            _signal = numpy.array([signal_dict[signal_id][:] for signal_id in signal_ids]).T
        except:
            _signal = list()
            for signal_id in signal_ids:
                if signal_id in signal_dict.keys():
                    _signal.append(signal_dict[signal_id][:])
                else:
                    sys.stderr.write(f'ERROR: signal id {signal_id} does not exist in {filename}\n')
                    sys.stderr.flush()
            _signal = numpy.array(_signal)
        data_pieces.append((_signal, 0))
    #
    return data_pieces


class RawDataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, batch_size = 20, window_length = 256 * 4, shift = 256 * 2,
                 do_shuffle = False,
                 do_standard_scaling = True,
                 mode = None,
                 patient_id = 'no_patient_id', load_stats = False,
                 debug_mode = False):
        '''

            Constructor to create objects of the class **RawDataGenerator** and load all the data.
            Future implementations will pave the way to load data from files on-the-fly in order to allow work
            with large enough datasets.

            Parameters
            ----------

            :param self:
                Reference to the current object.

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

            :param boolean exclude_seizures:
                Flag to indicate whether to exclude the records with seizures.

            :param string mode:
                String to indicate the current mode of the process: "train", "val", "test".
            
            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

            :param bool load_stats:
                Flag to indicate whether to load statistics from a file.

            :param bool debug_mode:
                Flag to indicate whether to activate debug mode.

        '''
        #
        self.batch_size = batch_size
        self.window_length = window_length
        self.shift = shift
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.mode = mode
        self.patient_id = patient_id
        self.load_stats = load_stats
        self.debug_mode = debug_mode
        #
        self.input_shape = None
        self.training_batches = 0
        self.validation_batches = 0
        self.num_training_samples = 0
        self.num_test_samples = 0
        self.num_validation_samples = 0
        self.num_signal_vectors = 0
        #
        self.train_indices = list()
        self.val_indices = list()
        self.test_indices = list()
        self.current_fold = -1
        #
        self.interictal_data = list()
        self.preictal_data = list()
        self.train_data = list()
        #self.validation_data = list()
        self.test_data = list()
        #

        #interictal = list()
        _input_shape = None
        print("Loading data...")
        self.patient_dir = 'data/' + self.patient_id + '_raw'
        self.interictal_data = numpy.load(os.path.join(self.patient_dir, self.patient_id + '_interictal_raw.npy'))# , mmap_mode='r')
        print(f'Interictal data shape: {self.interictal_data.shape}')


        for filename in os.listdir(self.patient_dir):
            if 'preictal' in filename:
                preictal = numpy.load(os.path.join(self.patient_dir, filename))
                self.preictal_data.append(preictal)
        #
                
        self.num_seizures = len(self.preictal_data)
        self.input_shape = (self.window_length,) + self.interictal_data[0].shape
        print("Signals loaded!")
        print("Number of seizures available:", self.num_seizures)
        if self.num_seizures < 3:
            raise Exception('Not enough seizures, please try other parameters or patients.')

        self.next_fold()
        #
        self.mean = numpy.zeros(self.input_shape[-1])
        self.std  = numpy.ones(self.input_shape[-1])
        #
        if self.do_standard_scaling:
            if self.mode == 'train' and not self.load_stats:
                print("Calculating mean and std for scaling the data...")
                means = []
                counts = []
                stddevs = []
                for p in tqdm(self.interictal_data):
                    means.append(p.mean(axis = 0))
                    counts.append(len(p))
                    stddevs.append(p.std(axis = 0))
                
                for seizure in tqdm(self.preictal_data):
                    for p in seizure:
                        means.append(p.mean(axis = 0))
                        counts.append(len(p))
                        stddevs.append(p.std(axis = 0))

                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)

                #
                array = numpy.array([self.mean, self.std])
                numpy.save(os.path.join(self.patient_dir, self.patient_id + '_raw_stats.npy'), array)
                del array
                del means
                del counts
                del stddevs
            else:
                print("Loading mean and std for scaling the data...")
                stats_file = os.path.join(self.patient_dir, self.patient_id + '_raw_stats.npy')
                self.mean, self.std = numpy.load(stats_file)
                #array = numpy.load('models/statistics_raw_' + self.patient_id + '.npy')
                #self.mean = array[0]
                #self.std = array[1]
        #
        self.on_epoch_end()

    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        if self.mode == 'train':
            return self.training_batches
        elif self.mode == 'val':
            return self.validation_batches
        elif self.mode == 'test':
            return self.test_batches

    def on_epoch_end(self):
        '''
        Performs the predefined actions after an epoch is complete.
        Currently only shuffles data if such option was enabled in the constructor.

        :param self: Reference to the current object.
        '''
        #
        self.training_batches = self.num_training_samples // self.batch_size
        if (self.num_training_samples % self.batch_size) != 0:
            self.training_batches += 1
        #
        self.validation_batches = self.num_validation_samples // self.batch_size
        if (self.num_validation_samples % self.batch_size) != 0:
            self.validation_batches += 1
        #
        self.test_batches = self.num_test_samples // self.batch_size
        if (self.num_test_samples % self.batch_size) != 0:
            self.test_batches += 1
        #

        if self.do_shuffle:
            shuffle(self.train_indices)
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
            idx = batch_index * self.batch_size + sample
            if self.mode == 'train':
                #index, label = self.train_indices[idx]
                index, label = self.train_indices[idx % len(self.train_indices)]
                X.append(numpy.array(self.train_data[index : index + self.window_length], dtype=numpy.float64))

            elif self.mode == 'val':
                #index, label = self.val_indices[batch_index * self.batch_size + sample]
                index, label = self.val_indices[idx % len(self.val_indices)]
                X.append(numpy.array(self.train_data[index : index + self.window_length], dtype=numpy.float64))

            elif self.mode == 'test':
                index, label = self.test_indices[idx % len(self.test_indices)]
                #index, label = self.test_indices[batch_index * self.batch_size + sample]
                X.append(numpy.array(self.test_data[index : index + self.window_length], dtype=numpy.float64))
            #
            Y.append(label)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)
        #
        if self.do_standard_scaling:
            for i in range(len(X)):
                X[i] = (X[i] - self.mean) / self.std
        #
        if X.min() < -10. or X.max() > 10.:
            if self.debug_mode:
                print("#  ", file = sys.stderr)
                print("#  WARNING: too large values after scaling while getting batch %d" % batch_index, file = sys.stderr)
                print("#  min value = %g" % X.min(), file = sys.stderr)
                print("#  max value = %g" % X.max(), file = sys.stderr)
            #
            X = numpy.minimum(X,  10.)
            X = numpy.maximum(X, -10.)
            if self.debug_mode:
                print("#  values of all the samples in this batch clipped, current limits are [%f, %f]" % (X.min(), X.max()), file = sys.stderr)
                print("#  ", file = sys.stderr)
        #
        return (X, Y)

    def next_fold(self):
        """
        Generates the next fold of training-validation-test sets.
        
        :param self: Reference to the current object.

        :return None
        """
        self.current_fold += 1
        if self.current_fold == self.num_seizures:
            raise Exception('No more folds available.')

        print('Generating fold %d of %d' % ((self.current_fold + 1), self.num_seizures))


        # Generate indices for training-test splits of current fold
        interictal_block_size = len(self.interictal_data) // self.num_seizures
        interictal_indices = [i for i in range(len(self.interictal_data))]
        idx = self.current_fold * interictal_block_size

        train_interictal_indices = list()
        train_preictal_indices = list()
        test_interictal_indices = list()
        test_preictal_indices = list()

        if self.current_fold == self.num_seizures - 1:
            test_interictal_indices = interictal_indices[idx:] # Last block of the fold may have more samples
            train_interictal_indices = interictal_indices[:idx]
        else:
            test_interictal_indices = interictal_indices[idx:(idx + interictal_block_size)]
            train_interictal_indices = interictal_indices[:idx] + interictal_indices[(idx + interictal_block_size):]

        for i in range(self.num_seizures):
            if i == self.current_fold:
                test_preictal_indices.append(i)
            else:
                train_preictal_indices.append(i)
        #
        self.train_data = list()
        for i in train_interictal_indices:
            self.train_data.append(self.interictal_data[i])
        first_train_preictal = len(self.train_data)
        for i in train_preictal_indices:
            for sample in self.preictal_data[i]:
                self.train_data.append(sample)

        self.test_data = list()
        for i in test_interictal_indices:
            self.test_data.append(self.interictal_data[i])
        first_test_preictal = len(self.test_data)
        for i in test_preictal_indices:
            for sample in self.preictal_data[i]:
                self.test_data.append(sample)
        #

        self.train_indices = list()
        self.val_indices = list()
        self.test_indices = list()

        #######################################################################
        # Generate train indices
        interictal_train_samples = 0
        for i in numpy.arange(0, first_train_preictal - self.window_length, step = self.shift):
            self.train_indices.append((i, 0))
            interictal_train_samples += 1
        
        i = int(interictal_train_samples * 0.80)
        interictal_train_samples -= i
        self.val_indices = self.train_indices[i:] # Last 20% of the samples for validation
        self.train_indices = self.train_indices[:i]

        
        # Oversampling on preictal samples
        preictal_train_samples = 0
        for i in numpy.arange(first_train_preictal, len(self.train_data) - self.window_length, step = self.shift // 5):
            self.train_indices.append((i, 1))
            preictal_train_samples += 1
        
        i = interictal_train_samples + int(preictal_train_samples * 0.80)
        preictal_train_samples = preictal_train_samples - i + interictal_train_samples
        self.val_indices = self.val_indices + self.train_indices[i:] # Last 20% of the samples for validation
        self.train_indices = self.train_indices[:i]
        #######################################################################
        
        #######################################################################
        # Generate test indices
        interictal_test_samples = 0
        for i in numpy.arange(0, first_test_preictal - self.window_length, step = self.shift):
            self.test_indices.append((i, 0))
            interictal_test_samples += 1
        
        # NOT oversampling on TEST
        preictal_test_samples = 0
        for i in numpy.arange(first_test_preictal, len(self.test_data) - self.window_length, step = self.shift):
            self.test_indices.append((i, 1))
            preictal_test_samples += 1
        #######################################################################

        self.num_training_samples = interictal_train_samples + preictal_train_samples
        self.num_test_samples = interictal_test_samples + preictal_test_samples
        self.num_validation_samples = len(self.val_indices)
        

        print("Fold generated")
        print("Input shape: ", self.input_shape)
        interictal_train_samples = interictal_train_samples / 0.80
        preictal_train_samples = preictal_train_samples / 0.80
        print("Training 80%% + Val 20%% interictal samples: %d (%0.2f %%)" % (interictal_train_samples, 100.0 * interictal_train_samples / (interictal_train_samples + preictal_train_samples)))
        print("Training 80%% + Val 20%% preictal samples: %d (%0.2f %%)" % (preictal_train_samples, 100.0 * preictal_train_samples / (interictal_train_samples + preictal_train_samples)))
        print("Test interictal samples: %d (%0.2f %%)" % (interictal_test_samples, 100.0 * interictal_test_samples / (interictal_test_samples + preictal_test_samples)))
        print("Test preictal samples: %d (%0.2f %%)" % (preictal_test_samples, 100.0 * preictal_test_samples / (interictal_test_samples + preictal_test_samples)))


class RawDataProcessor:
    '''
        Class for preprocessing the EEG raw signals and create a dataset.
    '''

    def __init__(self, index_filenames,
                 min_interictal_length = 256 * 3600 * 4, # Select interictal samples with at least 4h of interictal period
                 preictal_length = 256 * 3600, # 1 hour before the seizure
                 do_preemphasis = False,
                 exclude_seizures = False,
                 patient_id = 'no_patient_id'):
        '''

            Constructor to create objects of the class **RawDataProcessor** and load all the data.
            Future implementations will pave the way to load data from files on-the-fly in order to allow work
            with large enough datasets.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int min_interictal_length:
                Minimum number of vectors of channels in a period to consider
                the period as interictal.

            :param int preictal_length:
                Length of the preictal period in number of vectors.

            :param boolean do_preemphasis:
                Flag to indicate whether to apply a preemphasis FIR filter
                to each signal/channel.

            :param boolean exclude_seizures:
                Flag to indicate whether to exclude the records with seizures.
            
            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

        '''
        #
        self.do_preemphasis = do_preemphasis
        self.exclude_seizures = exclude_seizures
        self.min_interictal_length = min_interictal_length
        self.preictal_length = preictal_length
        self.patient_id = patient_id
        #
        self.input_shape = None
        self.num_signal_vectors = 0
        #
        os.makedirs('data', exist_ok = True)
        if os.path.exists('data/' + self.patient_id + '_raw'):
            if len(os.listdir('data/' + self.patient_id + '_raw')) > 0:
                raise Exception('Found already processed data from patient ' + self.patient_id 
                                + ', please clean the following dir: \n' + 'data/' + self.patient_id + '_raw')
            #
        #
        else:
            os.makedirs('data/' + self.patient_id + '_raw')
            #os.makedirs('data/' + self.patient_id + '/interictal', exist_ok = True)
            #os.makedirs('data/' + self.patient_id + '/preictal', exist_ok = True)
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    if l[0] != '#':
                        self.filenames.append(l.strip() + '.edf.pbz2')
                f.close()
        #
        self.interictal_data = list()
        self.preictal_data = list()
        #

        interictal = list()
        print("Loading EDF signals...")
        for fname in tqdm(self.filenames):
            d_p = load_file(fname, exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = True,
                        verbose = 0)

            for p, label in d_p:

                if label == 0:
                    for x in p:
                        interictal.append(numpy.array(x, dtype=numpy.float64))
                        #if self.input_shape is None:
                        #    self.input_shape = (self.window_length,) + x.shape
                #
                elif label == 1:
                    if len(interictal) > (256 * 3600 // 2):
                        # More than half an hour from the previous seizure, consider them as separated seizures
                        preictal_len = min(self.preictal_length, len(interictal))
                        preictal = interictal[(len(interictal) - preictal_len):]
                        self.preictal_data.append(numpy.array(preictal, dtype=numpy.float64))
                        self.num_signal_vectors += len(preictal)

                        if (len(interictal) - preictal_len) >= self.min_interictal_length:
                            # Enough interictal data, store it
                            interictal = interictal[:(len(interictal) - preictal_len)]
                            self.interictal_data.append(numpy.array(interictal, dtype=numpy.float64))
                            self.num_signal_vectors += len(interictal)
                    #   
                    interictal = list() # Reset interictal
                #
            #
            if len(interictal) >= self.min_interictal_length:
                self.interictal_data.append(numpy.array(interictal, dtype=numpy.float64))
                interictal = list() # Reset interictal
        #       
        
        self.num_seizures = len(self.preictal_data)
        print("Signals loaded!")
        print("Number of seizures available:", self.num_seizures)
        if self.num_seizures < 3:
            raise Exception('Not enough seizures, please try other parameters or patients.')

        self.interictal_data = numpy.vstack(self.interictal_data)
        #

        # Save the data
        numpy.save('data/' + self.patient_id + '_raw/' + self.patient_id + '_interictal_raw.npy', self.interictal_data)

        for index in range(len(self.preictal_data)):
            preictal_period = self.preictal_data[index]
            dest_filename = 'data/' + self.patient_id + '_raw/' + self.patient_id + '_preictal_raw_' + str(index) + '.npy'
            numpy.save(dest_filename, numpy.array(preictal_period, dtype=numpy.float64))
        #


class EEGDataProcessor:
    """
        Class for processing EEG signals, extracting spectral and time domain features.
    """
    def __init__(self, index_filenames, window_length = 256 * 10, shift = 256 * 10,
                min_interictal_length = 256 * 3600 * 4, # Select interictal samples with at least 4h of interictal period
                preictal_length = 256 * 3600, # 1 hour before the seizure
                do_preemphasis = False,
                exclude_seizures = False,
                patient_id = 'no_patient_id'):
        '''

                Constructor to create objects of the class **EEGDataProcessor** and load all the data.
                Future implementations will pave the way to load data from files on-the-fly in order to allow work
                with large enough datasets.

                Parameters
                ----------

                :param self:
                    Reference to the current object.

                :param list index_filenames:
                    List of index filenames from which to load data filenames.

                :param int min_interictal_length:
                    Minimum number of vectors of channels in a period to consider
                    the period as interictal.

                :param int preictal_length:
                    Length of the preictal period in number of vectors.

                :param boolean do_preemphasis:
                    Flag to indicate whether to apply a preemphasis FIR filter
                    to each signal/channel.

                :param boolean exclude_seizures:
                    Flag to indicate whether to exclude the records with seizures.
                
                :param string patient_id:
                    String to indicate the patient id. It is used to save the statistics file.

        '''

        #
        self.window_length = window_length
        self.shift = shift
        self.do_preemphasis = do_preemphasis
        self.exclude_seizures = exclude_seizures
        self.min_interictal_length = min_interictal_length
        self.preictal_length = preictal_length
        self.patient_id = patient_id
        #
        os.makedirs('data', exist_ok = True)
        dataset_dir = 'data/' + self.patient_id + '_processed'
        if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            raise Exception('Found already processed data from patient ' + self.patient_id 
                            + ', please clean the following dir: \n' + dataset_dir)
            #
        #
        else:
            os.makedirs('data/' + self.patient_id + '_processed', exist_ok = True)
            os.makedirs(dataset_dir, exist_ok = True)
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    if l[0] != '#':
                        self.filenames.append(l.strip() + '.edf.pbz2')
                f.close()
        #
        #
        self.interictal_data = list()
        self.preictal_data = list()
        #

        interictal = list()
        print("Loading EDF signals...")
        for fname in tqdm(self.filenames):
            d_p = load_file(fname, exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = True,
                        verbose = 0)

            for p, label in d_p:

                if label == 0:
                    for x in p:
                        interictal.append(numpy.array(x, dtype=numpy.float64))
                #
                elif label == 1:
                    if len(interictal) > (256 * 3600 // 2):
                        # More than half an hour from the previous seizure, consider them as separated seizures
                        preictal_len = min(self.preictal_length, len(interictal))
                        preictal = interictal[(len(interictal) - preictal_len):]
                        self.preictal_data.append(numpy.array(preictal, dtype=numpy.float64))

                        if (len(interictal) - preictal_len) >= self.min_interictal_length:
                            # Enough interictal data, store it
                            interictal = interictal[:(len(interictal) - preictal_len)]
                            self.interictal_data.append(numpy.array(interictal, dtype=numpy.float64))
                    #   
                    interictal = list() # Reset interictal
                #
            #
            if len(interictal) >= self.min_interictal_length:
                self.interictal_data.append(numpy.array(interictal, dtype=numpy.float64))
                interictal = list() # Reset interictal
        #
        self.num_seizures = len(self.preictal_data)
        print("Signals loaded!")
        print("Number of seizures available:", self.num_seizures)
        if self.num_seizures < 3:
            raise Exception('Not enough seizures, please try other parameters or patients.')


        # Extract features
        n_channels = 21 # Warning: This is set manually

        # Interictal preprocessors
        self.interictal_preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                subsampling_period = self.shift // 256 * 1000, # in ms, self.shift is provided in seconds
                                                window_length = self.window_length // 256 * 1000, # in ms
                                                fb_length = 20, # number of filters
                                                use_mel_scale = False,
                                                use_eeg_filtering = True,
                                                max_freq_for_filters = 70)
                    for _ in range(n_channels)]


        # Oversampling preprocessors for preictal
        self.preictal_preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                # Oversampling on preictal data, self.shift is provided in seconds
                                                subsampling_period = self.shift // 256 * 1000 // 5, # in ms
                                                window_length = self.window_length // 256 * 1000, # in ms
                                                fb_length = 20, # number of filters
                                                use_mel_scale = False,
                                                use_eeg_filtering = True,
                                                max_freq_for_filters = 70)
                    for _ in range(n_channels)]

        # Normal preprocessors for preictal
        self.preictal_normal_preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                subsampling_period = self.shift // 256 * 1000, # in ms
                                                window_length = self.window_length // 256 * 1000, # in ms
                                                fb_length = 20, # number of filters
                                                use_mel_scale = False,
                                                use_eeg_filtering = True,
                                                max_freq_for_filters = 70)
                    for _ in range(n_channels)]
        

        #############################################
        # Interictal signal
        #############################################
        print('Processing interictal data...')
        self.processing_interictal = True

        channel_list = [x for x in range(n_channels)]

        fbank = list()
        time_domain_statistics = list()

        for ch in channel_list:
            fbank.append([])
            time_domain_statistics.append([])

        
        for interictal_period in tqdm(self.interictal_data):

            # Params for multiprocessing
            params = zip(channel_list, [interictal_period[:,ch] for ch in channel_list])

            with Pool(cpu_count()) as pool:
                pool_output = pool.starmap(self.process_channel, params)
                for po in pool_output:
                    ch = po[0]
                    fbank[ch].append(po[1])
                    time_domain_statistics[ch].append(po[2])

        num_samples = sum([len(x) for x in fbank[0]])
        X = numpy.zeros([num_samples, len(channel_list), fbank[0][0].shape[1] + time_domain_statistics[0][0].shape[1]])

        # Fill the X array with fbank and statistics values
        # Shape: (num_samples, num_channels, 8+6),  fbank+stats = 8+6 features
        for ch in range(len(channel_list)):
            i = 0
            for x in fbank[ch]:
                X[i : i + len(x), ch, :x.shape[1]] = x
            #
            x_shape_1 = fbank[ch][0].shape[1]
            for s in time_domain_statistics[ch]:
                X[i : i + len(s), ch, x_shape_1:] = s
            #
            i += len(fbank[ch][0])

        print(f'Interictal array shape: {X.shape}')
        numpy.save(dataset_dir + '/' + self.patient_id + '_interictal.npy', X)

        # Free some memory
        del self.interictal_data


        ######################################################################################
        # Preictal signal
        # We will process preictal two times.
        # 1.- Oversampling with a smaller shift of the window to have more training samples
        # 2.- Not oversampling to get the test samples
        ######################################################################################

        print('Processing preictal data oversampling...')
        self.processing_interictal = False

        # 1.- Oversampling 
        self.oversample = True
        num_preictal_samples = 0

        for index in tqdm(range(len(self.preictal_data))):
            
            preictal_period = self.preictal_data[index]
            #print(f'Preictal period length: {len(preictal_period)}')
            fbank = list()
            time_domain_statistics = list()
            for ch in channel_list:
                fbank.append([])
                time_domain_statistics.append([])

            # Params for multiprocessing
            params = zip(channel_list, [preictal_period[:,ch] for ch in channel_list])

            with Pool(cpu_count()) as pool:
                pool_output = pool.starmap(self.process_channel, params)
                #pool_output = pool.starmap(process_channel, channel_list)
                for po in pool_output:
                    ch = po[0]
                    fbank[ch].append(po[1])
                    time_domain_statistics[ch].append(po[2])
            #
            num_samples = sum([len(x) for x in fbank[0]])
            print(f'Num samples: {num_samples}')
            X = numpy.zeros([num_samples, len(channel_list), fbank[0][0].shape[1] + time_domain_statistics[0][0].shape[1]])
        
            # Fill the X array with fbank and statistics values
            # Shape: (num_samples, num_channels, 8+6),  fbank+stats = 8+6 features
            for ch in range(len(channel_list)):
                i = 0
                for x in fbank[ch]:
                    X[i : i + len(x), ch, :x.shape[1]] = x
                #
                x_shape_1 = fbank[ch][0].shape[1]
                for s in time_domain_statistics[ch]:
                    X[i : i + len(s), ch, x_shape_1:] = s
                #
                i += len(fbank[ch][0])
            #
            num_preictal_samples += len(X)
            numpy.save(dataset_dir + '/' + self.patient_id + '_preictal_' + str(index) + '.npy', X)
        #
        print(f'Preictal num samples (oversampling): {num_preictal_samples}')
        ##

        # 2.- Normal preprocess
        print('Processing preictal data not oversampling...')
        self.oversample = False
        num_preictal_samples = 0

        for index in tqdm(range(len(self.preictal_data))):
            
            preictal_period = self.preictal_data[index]
            #print(f'Preictal period length: {len(preictal_period)}')
            fbank = list()
            time_domain_statistics = list()
            for ch in channel_list:
                fbank.append([])
                time_domain_statistics.append([])

            # Params for multiprocessing
            params = zip(channel_list, [preictal_period[:,ch] for ch in channel_list])

            with Pool(cpu_count()) as pool:
                pool_output = pool.starmap(self.process_channel, params)
                #pool_output = pool.starmap(process_channel, channel_list)
                for po in pool_output:
                    ch = po[0]
                    fbank[ch].append(po[1])
                    time_domain_statistics[ch].append(po[2])
            #
            num_samples = sum([len(x) for x in fbank[0]])
            print(f'Num samples: {num_samples}')
            X = numpy.zeros([num_samples, len(channel_list), fbank[0][0].shape[1] + time_domain_statistics[0][0].shape[1]])
        
            # Fill the X array with fbank and statistics values
            # Shape: (num_samples, num_channels, 8+6),  fbank+stats = 8+6 features
            for ch in range(len(channel_list)):
                i = 0
                for x in fbank[ch]:
                    X[i : i + len(x), ch, :x.shape[1]] = x
                #
                x_shape_1 = fbank[ch][0].shape[1]
                for s in time_domain_statistics[ch]:
                    X[i : i + len(s), ch, x_shape_1:] = s
                #
                i += len(fbank[ch][0])
            #
            num_preictal_samples += len(X)
            numpy.save(dataset_dir + '/' + self.patient_id + '_preictal_oversampled_' + str(index) + '.npy', X)
        #
        print(f'Preictal num samples (normal preprocess): {num_preictal_samples}')
        ##
        

    def process_channel(self, ch, period):
        if self.processing_interictal:
            mss = preprocessor.MySignalStats(period, window_length = self.window_length, subsampling_period = self.shift)
            self.interictal_preprocessors[ch].preemphasis_alpha = 0.50
            preemphasis, spectrogram, fb, fb_choi, mfcc = self.interictal_preprocessors[ch].preprocess_an_utterance(mss, verbose = 0)
        else:
            if self.oversample:
                mss = preprocessor.MySignalStats(period, window_length = self.window_length, subsampling_period = self.shift // 5)
                self.preictal_preprocessors[ch].preemphasis_alpha = 0.50
                preemphasis, spectrogram, fb, fb_choi, mfcc = self.preictal_preprocessors[ch].preprocess_an_utterance(mss, verbose = 0)
            else:
                mss = preprocessor.MySignalStats(period, window_length = self.window_length, subsampling_period = self.shift)
                self.preictal_normal_preprocessors[ch].preemphasis_alpha = 0.50
                preemphasis, spectrogram, fb, fb_choi, mfcc = self.preictal_normal_preprocessors[ch].preprocess_an_utterance(mss, verbose = 0)
            #
        #

        assert len(fb) == len(mss.time_domain_statistics), f'{len(fb)}, {len(mss.time_domain_statistics)}'

        return (ch, fb, mss.time_domain_statistics)


class EEGDataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, batch_size = 20, #window_length = 256 * 4, shift = 256 * 2,
                 do_shuffle = False,
                 do_standard_scaling = True,
                 mode = None,
                 patient_id = 'no_patient_id', load_stats = False,
                 debug_mode = False):
        '''

            Constructor to create objects of the class **RawDataGenerator** and load all the data.
            Future implementations will pave the way to load data from files on-the-fly in order to allow work
            with large enough datasets.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param int batch_size:
                Number of samples per batch.

            :param boolean do_shuffle:
                Flag to indicate whether to shuffle data between epochs.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale each channel to zero
                mean and unit variance.

            :param boolean do_preemphasis:
                Flag to indicate whether to apply a preemphasis FIR filter
                to each signal/channel.

            :param boolean exclude_seizures:
                Flag to indicate whether to exclude the records with seizures.

            :param string mode:
                String to indicate the current mode of the process: "train", "val", "test".
            
            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

            :param bool load_stats:
                Flag to indicate whether to load statistics from a file.

            :param bool debug_mode:
                Flag to indicate whether to activate debug mode.

        '''
        #
        self.batch_size = batch_size
        #self.window_length = window_length
        #self.shift = shift
        self.do_shuffle = do_shuffle
        self.do_standard_scaling = do_standard_scaling
        self.mode = mode
        self.patient_id = patient_id
        self.load_stats = load_stats
        self.debug_mode = debug_mode
        #
        self.input_shape = None
        self.training_batches = 0
        self.validation_batches = 0
        self.num_training_samples = 0
        self.num_test_samples = 0
        self.num_validation_samples = 0
        #
        self.train_indices = list()
        self.val_indices = list()
        self.test_indices = list()
        self.current_fold = -1
        #
        self.interictal_data = list()
        self.preictal_data = list()
        self.preictal_normal = list()
        #
        self.train_data = list()
        self.test_data = list()
        #

        #interictal = list()
        _input_shape = None
        print("Loading data...")
        self.patient_dir = 'data/' + self.patient_id + '_processed'
        self.interictal_data = numpy.load(os.path.join(self.patient_dir, self.patient_id + '_interictal.npy'))# , mmap_mode='r')
        print(f'Interictal data shape: {self.interictal_data.shape}')


        for filename in os.listdir(self.patient_dir):
            #if 'preictal' in filename:
            if 'preictal_oversampled' in filename:
                preictal_oversampled = numpy.load(os.path.join(self.patient_dir, filename))
                fname = filename.replace('oversampled_', '')
                preictal_normal = numpy.load(os.path.join(self.patient_dir, fname))
                self.preictal_data.append(preictal_oversampled)
                self.preictal_normal.append(preictal_normal)
                #preictal = numpy.load(os.path.join(self.patient_dir, filename))
                #self.preictal_data.append(preictal)
        #
                
        self.num_seizures = len(self.preictal_data)
        self.input_shape = self.interictal_data[0].shape
        print("Signals loaded!")
        print("Number of seizures available:", self.num_seizures)
        if self.num_seizures < 3:
            raise Exception('Not enough seizures, please try other parameters or patients.')

        self.next_fold()
        #
        self.mean = numpy.zeros(self.input_shape[-1])
        self.std  = numpy.ones(self.input_shape[-1])
        #
        if self.do_standard_scaling:
            if self.mode == 'train' and not self.load_stats:
                print("Calculating mean and std for scaling the data...")
                means = []
                counts = []
                stddevs = []
                for p in tqdm(self.interictal_data):
                    means.append(p.mean(axis = 0))
                    counts.append(len(p))
                    stddevs.append(p.std(axis = 0))
                
                for seizure in tqdm(self.preictal_data):
                    for p in seizure:
                        means.append(p.mean(axis = 0))
                        counts.append(len(p))
                        stddevs.append(p.std(axis = 0))

                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)

                #
                array = numpy.array([self.mean, self.std])
                numpy.save(os.path.join(self.patient_dir, self.patient_id + '_stats.npy'), array)
                del array
                del means
                del counts
                del stddevs
            else:
                print("Loading mean and std for scaling the data...")
                stats_file = os.path.join(self.patient_dir, self.patient_id + '_stats.npy')
                self.mean, self.std = numpy.load(stats_file)
        #
        self.on_epoch_end()

    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        if self.mode == 'train':
            return self.training_batches
        elif self.mode == 'val':
            return self.validation_batches
        elif self.mode == 'test':
            return self.test_batches

    def on_epoch_end(self):
        '''
        Performs the predefined actions after an epoch is complete.
        Currently only shuffles data if such option was enabled in the constructor.

        :param self: Reference to the current object.
        '''
        #
        self.training_batches = self.num_training_samples // self.batch_size
        if (self.num_training_samples % self.batch_size) != 0:
            self.training_batches += 1
        #
        self.validation_batches = self.num_validation_samples // self.batch_size
        if (self.num_validation_samples % self.batch_size) != 0:
            self.validation_batches += 1
        #
        self.test_batches = self.num_test_samples // self.batch_size
        if (self.num_test_samples % self.batch_size) != 0:
            self.test_batches += 1
        #

        if self.do_shuffle:
            shuffle(self.train_indices)
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
            idx = batch_index * self.batch_size + sample
            if self.mode == 'train':
                #index, label = self.train_indices[idx]
                index, label = self.train_indices[idx % len(self.train_indices)]
                X.append(numpy.array(self.train_data[index], dtype=numpy.float64))

            elif self.mode == 'val':
                #index, label = self.val_indices[batch_index * self.batch_size + sample]
                index, label = self.val_indices[idx % len(self.val_indices)]
                X.append(numpy.array(self.train_data[index], dtype=numpy.float64))

            elif self.mode == 'test':
                index, label = self.test_indices[idx % len(self.test_indices)]
                #index, label = self.test_indices[batch_index * self.batch_size + sample]
                X.append(numpy.array(self.test_data[index], dtype=numpy.float64))
            #
            Y.append(label)
        #
        X = numpy.array(X)
        Y = numpy.array(Y)
        #
        if self.do_standard_scaling:
            for i in range(len(X)):
                X[i] = (X[i] - self.mean) / self.std
        #
        return (X, Y)

    def next_fold(self):
        """
        Generates the next fold of training-validation-test sets.
        
        :param self: Reference to the current object.

        :return None
        """
        self.current_fold += 1
        if self.current_fold == self.num_seizures:
            raise Exception('No more folds available.')

        print('Generating fold %d of %d' % ((self.current_fold + 1), self.num_seizures))


        # Generate indices for training-test splits of current fold
        interictal_block_size = len(self.interictal_data) // self.num_seizures
        interictal_indices = [i for i in range(len(self.interictal_data))]
        idx = self.current_fold * interictal_block_size

        
        # Train-Test Interictal indices
        if self.current_fold == self.num_seizures - 1:
            test_interictal_indices = interictal_indices[idx:] # Last block of the fold may have more samples
            train_interictal_indices = interictal_indices[:idx]
        else:
            test_interictal_indices = interictal_indices[idx:(idx + interictal_block_size)]
            train_interictal_indices = interictal_indices[:idx] + interictal_indices[(idx + interictal_block_size):]


        # Train-Test Preictal indices
        train_preictal_indices = list()
        test_preictal_indices = list()
        
        for i in range(self.num_seizures):
            if i == self.current_fold:
                test_preictal_indices.append(i)
            else:
                train_preictal_indices.append(i)

        #
        # Training and validation indices
        #
        self.train_indices = list()
        self.val_indices = list()
        #
        self.train_data = list()
        j = 0
        for i in train_interictal_indices:
            self.train_data.append(self.interictal_data[i])
            self.train_indices.append((j, 0))
            j += 1

        self.val_indices = self.train_indices[int(len(self.train_indices) * 0.80):] # Last 20% for validation
        self.train_indices = self.train_indices[:int(len(self.train_indices) * 0.80)]

        #first_train_preictal = len(self.train_indices)
        #n_preictal = 0

        aux_indices = list()
        for i in train_preictal_indices:
            for sample in self.preictal_data[i]:
                self.train_data.append(sample)
                aux_indices.append((j, 1))
                j += 1
                #n_preictal += 1
        
        self.val_indices = self.val_indices + aux_indices[int(len(aux_indices) * 0.80):] # Last 20% for validation
        self.train_indices = self.train_indices + aux_indices[:int(len(aux_indices) * 0.80)]
        #val_pos = first_train_preictal + int(n_preictal * 0.80)

        #self.val_indices = self.val_indices + self.train_indices[val_pos:]
        #self.train_indices = self.train_indices[:val_pos]
        #
        # Test indices
        #
        self.test_data = list()
        self.test_indices = list()
        j = 0
        for i in test_interictal_indices:
            self.test_data.append(self.interictal_data[i])
            self.test_indices.append((j,0))
            j += 1

        first_test_preictal = len(self.test_data)
        for i in test_preictal_indices:
            for sample in self.preictal_normal[i]:
            #for sample in self.preictal_data[i]:
                self.test_data.append(sample)
                self.test_indices.append((j,1))
                j += 1
        #

        self.num_training_samples = len(self.train_indices)
        self.num_test_samples = len(self.test_indices)
        self.num_validation_samples = len(self.val_indices)
        

        print("Fold generated")
        print("Input shape: ", self.input_shape)
        interictal_train_samples = len(train_interictal_indices)
        preictal_train_samples = len(self.train_data) - interictal_train_samples
        interictal_test_samples = len(test_interictal_indices)
        preictal_test_samples = len(self.test_data) - interictal_test_samples
        print("Training 80%% + Val 20%% interictal samples: %d (%0.2f %%)" % (interictal_train_samples, 100.0 * interictal_train_samples / (interictal_train_samples + preictal_train_samples)))
        print("Training 80%% + Val 20%% preictal samples: %d (%0.2f %%)" % (preictal_train_samples, 100.0 * preictal_train_samples / (interictal_train_samples + preictal_train_samples)))
        print("Test interictal samples: %d (%0.2f %%)" % (interictal_test_samples, 100.0 * interictal_test_samples / (interictal_test_samples + preictal_test_samples)))
        print("Test preictal samples: %d (%0.2f %%)" % (preictal_test_samples, 100.0 * preictal_test_samples / (interictal_test_samples + preictal_test_samples)))

        print(f'Len val indices {len(self.val_indices)}')
        print(f'Len train indices {len(self.train_indices)}')
        print(f'Len test indices {len(self.test_indices)}')
        print(f'Len aux indices {len(aux_indices)}')


# ---------------------------------------------------------------------------



if __name__ == '__main__':

    #index_filenames = list()
    ##
    #for i in range(1, len(sys.argv)):
    #    if sys.argv[i] == '--index':
    #        index_filenames.append(sys.argv[i+1])
    ##
    #if index_filenames is None  or  len(index_filenames) == 0:
    #    raise Exception('Nothing can be done without data, my friend!')

    #dg = RawDataGenerator(index_filenames, mode = 'train', window_length = 256 * 10, shift = 256 * 5, do_shuffle = True)
    dg = EEGDataGenerator(mode = 'train', do_shuffle = True, patient_id = 'chb02')
    print(len(dg))
    for i in range(len(dg)):
        x, y = dg[i]
        print(x.shape, y.shape, i, len(dg))
    #print("loaded %d signal vectors" % dg.num_signal_vectors)
    print("available %d batches" % len(dg))

   
