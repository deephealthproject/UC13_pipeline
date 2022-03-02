"""
This script contains generator objects in order to perform epilepsy detection
in the Use Case 13 of DeepHealth project.
"""
import gc
import os
import numpy
from random import shuffle
from tqdm import tqdm
from file_utils import load_file


#-------------------------------------------------------------------------------


class RawRecurrentDataGenerator:
    '''
        Class for preprocessing the EEG raw signals.
    '''

    def __init__(self, index_filenames,
                 window_length = 1, # in seconds
                 shift = 0.5, # in seconds
                 timesteps = 19, # in seconds
                 sampling_rate = 256, # in Hz
                 batch_size = 10,
                 do_standard_scaling = True,
                 in_training_mode = False,
                 balance_batches = False,
                 patient_id = None):
        '''

            Constructor to create objects of the class 
            **RawRecurrentDataGenerator** and load all the data.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int window_length:
                Length of the window (in seconds) to slide through the signal.

            :param int shift:
                Number of seconds to shift the window to get each sample.

            :param int timesteps:
                Number of time steps for the sequences.

            :param int sampling_rate:
                Sampling rate of the signals, in Hz.

            :param int batch_size:
                Size of the batch to use.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale features to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

            :param boolean balance_batches:
                Flag to indicate whether to balance the batches.
            
            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

        '''
        #

        if patient_id is None:
            raise Exception('You have to specify a patient id, i.e. "chb01"')

        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
        self.balance_batches = balance_batches
        self.patient_id = patient_id
        #
        self.window_length = int(window_length * sampling_rate)
        self.sample_shift = int(sampling_rate * shift) # Half of sample length
        self.timesteps = timesteps
        self.num_channels = 23
        #
        self.input_shape = None
        self.num_batches = 0
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    if l[0] != '#':
                        self.filenames.append(l.strip() + '.edf.pbz2')
                f.close()
        #
        # List to store the data separated by files
        self.data = list()
        # List to store the labels associated at each sample separated by files
        self.labels = list()

        self.num_seizures = 0

        num_ictal = 0
        num_interictal = 0

        print('Loading EDF signals...')
        for i in tqdm(range(len(self.filenames))):
            d_p = load_file(self.filenames[i],
                            exclude_seizures = False,
                            do_preemphasis = False,
                            separate_seizures = True,
                            verbose = 0)

            len_file = 0

            self.data.append([])
            self.labels.append([])
            for p, label in d_p:
                self.data[-1].append(p) # p.tolist() # Append data
                self.labels[-1] += [label] * len(p)
                if label == 1:
                    self.num_seizures += 1
                    num_ictal += len(p)
                elif label == 0:
                    num_interictal += len(p)
                len_file += len(p)
            #
            #print(self.data[-1][0].shape)
            self.data[-1] = numpy.concatenate(self.data[-1], axis=0)
            #print(self.data[-1].shape)
        #
        # Iterate over the data list and generate indices for the sequences
        # We will treat each recording as independent from the others
        self.file_indexes = list()
        for i in range(len(self.data)):
            limit = len(self.data[i]) // self.sample_shift
            limit -= self.timesteps + 1
            limit = (limit * self.sample_shift) # - self.sample_shift 
            self.file_indexes.append(numpy.arange(0, limit, step = self.sample_shift).tolist())
        #     


        num_sequences_per_class = [0, 0]
        # Sequences indexes
        if self.balance_batches:
            # The batches will have the same samples for each class
            # List for the indices of class 0
            self.sequences_indexes_c0 = list()
            # List for the indices of class 1
            self.sequences_indexes_c1 = list()

            for fi in range(len(self.file_indexes)):
                for t in self.file_indexes[fi]:

                    label = self.labels[fi][t + (self.timesteps + 1) * self.sample_shift - 1]

                    if label == 0:
                        self.sequences_indexes_c0.append((fi, t))
                    elif label == 1:
                        self.sequences_indexes_c1.append((fi, t))
                #
            #
            
            num_sequences_per_class[0] = len(self.sequences_indexes_c0)
            num_sequences_per_class[1] = len(self.sequences_indexes_c1)
            num_sequences = num_sequences_per_class[0] + num_sequences_per_class[1]
            self.num_batches = num_sequences // self.batch_size
            if num_sequences % self.batch_size != 0:
                self.num_batches += 1
            #

        else:
            # Just one list for all of the sequences. No balancing
            self.sequences_indexes = list()

            for fi in range(len(self.file_indexes)):
                for t in self.file_indexes[fi]:
                    
                    label = self.labels[fi][t + (self.timesteps + 1) * self.sample_shift - 1]
                    num_sequences_per_class[label] += 1

                    self.sequences_indexes.append((fi, t, label))
                #
            #
            num_sequences = len(self.sequences_indexes)
            self.num_batches = num_sequences // self.batch_size
            if num_sequences % self.batch_size != 0:
                self.num_batches += 1
        #


        self.input_shape = (self.window_length, )

        print('Signals loaded!')
        print('\n-----------------------------------------------------------\n')
        print(f'Number of seizures available: {self.num_seizures}')

        print(f'Number of samples (not sequences): {num_ictal + num_interictal}')
        print(f'Interictal samples: {num_interictal} ({(num_interictal / (num_ictal + num_interictal) * 100):.2f} %)')
        print(f'Ictal samples: {num_ictal} ({(num_ictal / (num_ictal + num_interictal) * 100):.2f} %)')

        print(f'\nNumber of sequences: {num_sequences}')
        print(f'Interictal sequences: {num_sequences_per_class[0]} ({num_sequences_per_class[0] / num_sequences * 100.0 :.2f}%)')
        print(f'Ictal sequences: {num_sequences_per_class[1]} ({num_sequences_per_class[1] / num_sequences * 100.0 :.2f}%)')
        print(f'Number of batches: {self.num_batches}')
        print('\n-----------------------------------------------------------\n')


        # Standard scaling
        if self.do_standard_scaling:
            
            os.makedirs('stats', exist_ok=True)

            if self.in_training_mode:
                print('Calculating statistics to scale the data...')
                means = []
                counts = []
                stddevs = []
                for p in tqdm(self.data):
                    #p = numpy.array(p)
                    #print(p.shape)
                    means.append(p.mean())
                    counts.append(len(p))
                    stddevs.append(p.std())
                    #
                #
                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)
                #
                array = numpy.array([self.mean, self.std])
                numpy.save(f'stats/statistics_detection_raw_{patient_id}.npy', array)
                del means
                del counts
                del stddevs
                
            #
            else:
                print('Loading statistics to scale the data...')
                array = numpy.load(f'stats/statistics_detection_raw_{patient_id}.npy')
                self.mean = array[0]
                self.std = array[1]
            #
            del array
            gc.collect()
        #


    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        
        return self.num_batches
    

    def shuffle_data(self):
        '''
        Shuffles the data.

        :param self: Reference to the current object.
        '''
        #

        if self.balance_batches:
            shuffle(self.sequences_indexes_c0)
            shuffle(self.sequences_indexes_c1)
        else:
            shuffle(self.sequences_indexes)
        #


    def __getitem__(self, batch_index : int):
        '''
        Returns the batch of samples specified by the index.

        :param self: Reference to the current object.

        :param int batch_index: Index of the batch to retrieve.

        :return: A tuple with two objects, a batch of samples and the corresponding labels.
        '''
        
        X = list()
        Y = list()

        if self.balance_batches:

            half_batch = self.batch_size // 2

            # Class 0 sequences of samples
            for sequence in range(half_batch):

                idx = batch_index * self.batch_size + sequence
                idx = idx % len(self.sequences_indexes_c0)

                sequence_samples = list()

                # 'fi' is the index of the file in self.data, self.file_indexes and self.labels
                fi, t = self.sequences_indexes_c0[idx]

                

                for i in numpy.arange(t, t + (self.timesteps + 1) * self.sample_shift, step = self.sample_shift):
                    sequence_samples.append(self.data[fi][i : i + self.sampling_rate, :])
                
                label = 0 # It will always be 0 here

                X.append(sequence_samples)
                Y.append(label)

            # Class 1 sequences of samples
            for sequence in range(half_batch):
                
                idx = batch_index * self.batch_size + sequence
                idx = idx % len(self.sequences_indexes_c1)

                sequence_samples = list()

                # 'fi' is the index of the file in self.data, self.file_indexes and self.labels
                fi, t = self.sequences_indexes_c1[idx]

                for i in numpy.arange(t, t + (self.timesteps + 1) * self.sample_shift, step = self.sample_shift):
                    sequence_samples.append(self.data[fi][i : i + self.sampling_rate, :])
                
                label = 1 # It will always be 1 here

                X.append(sequence_samples)
                Y.append(label)

        else:
            # Do not balance batches
            for sequence in range(self.batch_size):
                idx = batch_index * self.batch_size + sequence

                sequence_samples = list()
                # 'fi' is the index of the file in self.data, self.file_indexes and self.labels
                
                if self.in_training_mode:
                    # Fill last batch with samples that have already been passed
                    # through the net in the same epoch
                    idx = idx % len(self.sequences_indexes)
                    fi, t, label = self.sequences_indexes[idx]

                    for i in numpy.arange(t, t + (self.timesteps + 1) * self.sample_shift, step = self.sample_shift):
                        sequence_samples.append(self.data[fi][i : i + self.sampling_rate, :])
                
                else:
                    # Do not drop last batch. It will have less samples than batch-size
                    # This will be done on validation and test
                    if idx < len(self.sequences_indexes):
                        fi, t, label = self.sequences_indexes[idx]

                        for i in numpy.arange(t, t + (self.timesteps + 1) * self.sample_shift, step = self.sample_shift):
                            sequence_samples.append(self.data[fi][i : i + self.sampling_rate, :])
                #

                if len(sequence_samples) > 0:
                    X.append(sequence_samples)
                    Y.append(label)
            #

        #
        X = numpy.array(X, dtype=numpy.float64)
        Y = numpy.array(Y, dtype=numpy.float64)

        if self.do_standard_scaling:
            X = (X - self.mean) / self.std
        
        if X.min() < -20. or X.max() > 20.:
            X = numpy.minimum(X,  20.)
            X = numpy.maximum(X, -20.)
        #
        Y = numpy.reshape(Y, (Y.shape[0], 1))

        return X, Y


#-------------------------------------------------------------------------------


class RawDataGenerator:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, index_filenames,
                 window_length = 1, # in seconds
                 shift = 0.5, # in seconds
                 sampling_rate = 256, # in Hz
                 batch_size=20,
                 do_standard_scaling=True,
                 in_training_mode=False,
                 balance_batches=False,
                 patient_id=None):
        '''

            Constructor to create objects of the class **DataGenerator** and
            load all the data.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int window_length:
                Size in vectors of channels in the signal to compose a
                single sample.

            :param int shift:
                Number of vectors of channels from the signal to shift from
                one sample to the next one in the sequence.

            :param int sampling_rate:
                Sampling rate of the signals, in Hz.

            :param int batch_size:
                Number of samples per batch.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale each channel to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

        '''
        #
        self.batch_size = batch_size
        self.window_length = int(window_length * sampling_rate)
        self.shift = int(shift * sampling_rate)
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
        self.balance_batches = balance_batches
        self.patient_id = patient_id
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    if l[0] != '#':
                        self.filenames.append(l.strip() + '.edf.pbz2')
                f.close()
        #
        self.indices = list()
        self.indices_ictal = list()
        self.data_pieces = list()
        data_pieces_index = 0

        self.num_samples = [0, 0] # [class_0, class_1]

        print('Loading EDF signals...')
        for file_index in tqdm(range(len(self.filenames))):
            recording = load_file(self.filenames[file_index],
                            exclude_seizures=False,
                            do_preemphasis=False,
                            separate_seizures=True,
                            verbose=0)
            
            if self.in_training_mode and self.balance_batches:
                # Separate data by label
                for values, label in recording:
                    self.data_pieces.append((values, label))

                    if label == 0:
                        for i in numpy.arange(0, len(values) - self.window_length + 1, step = self.shift):
                            self.indices.append((data_pieces_index, i))
                            self.num_samples[0] += 1
                    else:
                        for i in numpy.arange(0, len(values) - self.window_length + 1, step = self.shift):
                            self.indices_ictal.append((data_pieces_index, i))
                            self.num_samples[1] += 1
                    #
                    data_pieces_index += 1
                #
            else:
                # Do not separate data by label
                for values, label in recording:
                    self.data_pieces.append((values, label))

                    for i in numpy.arange(0, len(values) - self.window_length + 1, step = self.shift):
                        self.indices.append((data_pieces_index, i))
                        self.num_samples[label] += 1
                    #
                    data_pieces_index += 1
                #
            #
        #
        self.total_samples = sum(self.num_samples)
        self.num_batches = self.total_samples // self.batch_size
        if self.total_samples % self.batch_size != 0:
            self.num_batches += 1

        print(f'Total number of samples: {self.total_samples}')
        
        print(f'Interictal samples: {self.num_samples[0]} '
            + f'({self.num_samples[0] / self.total_samples * 100.0:.2f}%)')
        
        print(f'Ictal samples: {self.num_samples[1]} '
            + f'({self.num_samples[1] / self.total_samples * 100.0:.2f}%)')
        
        print(f'Number of batches: {self.num_batches}')

        #
        if self.do_standard_scaling:
            if self.in_training_mode:
                means = []
                counts = []
                stddevs = []
                for p, label in self.data_pieces:
                    means.append(p.mean())
                    counts.append(len(p))
                    stddevs.append(p.std())
                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)
                #
                array = numpy.array([self.mean, self.std])
                numpy.save(f'stats/raw_{self.patient_id}.npy', array)
                del means
                del counts
                del stddevs
            else:
                array = numpy.load(f'stats/raw_{self.patient_id}.npy')
                self.mean = array[0]
                self.std = array[1]
            #
            del array
        #


    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        return self.num_batches


    def shuffle_data(self):
        '''
        Shuffles the data.

        :param self: Reference to the current object.
        '''
        #
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
        if self.balance_batches:
            half_batch = self.batch_size // 2

            # Class interictal - 0
            for sample in range(half_batch):
                idx = (batch_index * self.batch_size + sample) % len(self.indices)
                
                data_pieces_index, index = self.indices[idx]
                data, label = self.data_pieces[data_pieces_index]

                X.append(data[index : index + self.window_length])
                Y.append(label)

            # Class ictal - 1
            for sample in range(half_batch):
                idx = (batch_index * self.batch_size + sample) % len(self.indices_ictal)
                
                data_pieces_index, index = self.indices_ictal[idx]
                data, label = self.data_pieces[data_pieces_index]

                X.append(data[index : index + self.window_length])
                Y.append(label)

        else:
            for sample in range(self.batch_size):
                idx = (batch_index * self.batch_size + sample)
                
                # Do not fill last batch with samples from the beginning
                if idx < len(self.indices):
                    data_pieces_index, index = self.indices[idx]
                    data, label = self.data_pieces[data_pieces_index]

                    X.append(data[index : index + self.window_length])
                    Y.append(label)
        #

        X = numpy.array(X)
        Y = numpy.array(Y)
        #
        if self.do_standard_scaling:
            X = (X - self.mean) / self.std
        #
        if X.min() < -20. or X.max() > 20.:
            X = numpy.minimum(X,  20.)
            X = numpy.maximum(X, -20.)
        #

        X = numpy.reshape(X, (X.shape[0], 1, )  + X.shape[1:])
        Y = numpy.reshape(Y, (Y.shape[0], 1))

        return (X, Y)


#-------------------------------------------------------------------------------


class RawDataGenerator2:
    '''
        Class for preloading data and provide batches.
    '''

    def __init__(self, index_filenames,
                 window_length = 1, # in seconds
                 shift = 0.5, # in seconds
                 sampling_rate = 256, # in Hz
                 batch_size=20,
                 do_standard_scaling=True,
                 in_training_mode=False,
                 balance_batches=False,
                 patient_id=None):
        '''

            Constructor to create objects of the class **DataGenerator2** and
            load all the data.

            This generator does not separate the signals in classes before
            extracting windows. There will be windows with samples of both
            classes at the onsets and offsets of the seizures. The label of
            an extracted window is the label of the last value inside it.

            Parameters
            ----------

            :param self:
                Reference to the current object.

            :param list index_filenames:
                List of index filenames from which to load data filenames.

            :param int window_length:
                Size in vectors of channels in the signal to compose a
                single sample.

            :param int shift:
                Number of vectors of channels from the signal to shift from
                one sample to the next one in the sequence.

            :param int sampling_rate:
                Sampling rate of the signals, in Hz.

            :param int batch_size:
                Number of samples per batch.

            :param boolean do_standard_scaling:
                Flag to indicate whether to scale each channel to zero
                mean and unit variance.

            :param boolean in_training_mode:
                Flag to indicate whether the process is in training mode.

            :param string patient_id:
                String to indicate the patient id. It is used to save the statistics file.

        '''
        #
        self.batch_size = batch_size
        self.window_length = int(window_length * sampling_rate)
        self.shift = int(shift * sampling_rate)
        self.do_standard_scaling = do_standard_scaling
        self.in_training_mode = in_training_mode
        self.balance_batches = balance_batches
        self.patient_id = patient_id
        #
        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    if l[0] != '#':
                        self.filenames.append(l.strip() + '.edf.pbz2')
                f.close()
        #
        self.indices = list()
        self.indices_ictal = list()
        self.data = list()

        self.num_samples = [0, 0] # [class_0, class_1]

        print('Loading EDF signals...')
        for file_index in tqdm(range(len(self.filenames))):

            recording = load_file(self.filenames[file_index],
                            exclude_seizures=False,
                            do_preemphasis=False,
                            separate_seizures=False,
                            verbose=0)

            values, labels = recording[0] # There is only 1 element on the list


            if self.in_training_mode and self.balance_batches:
                # Separate data by label
                self.data.append(values)

                for i in numpy.arange(0, len(values) - self.window_length + 1, step = self.shift):
                        label = labels[i + self.window_length - 1]
                        self.num_samples[label] += 1
                        if label == 0:
                            self.indices.append((file_index, i, label))
                        elif label == 1:
                            self.indices_ictal.append((file_index, i, label))
                        else:
                            raise Exception('Unexpected label')
                #
            else:
                # Do not separate data by label
                for values, labels in recording:
                    self.data.append(values)

                    for i in numpy.arange(0, len(values) - self.window_length + 1, step = self.shift):
                            label = labels[i + self.window_length - 1]
                            self.num_samples[label] += 1
                            self.indices.append((file_index, i, label))
                    #
                #
            #
        #

        self.total_samples = sum(self.num_samples)
        self.num_batches = self.total_samples // self.batch_size
        if self.total_samples % self.batch_size != 0:
            self.num_batches += 1

        print(f'Total number of samples: {self.total_samples}')
        
        print(f'Interictal samples: {self.num_samples[0]} '
            + f'({self.num_samples[0] / self.total_samples * 100.0:.2f}%)')
        
        print(f'Ictal samples: {self.num_samples[1]} '
            + f'({self.num_samples[1] / self.total_samples * 100.0:.2f}%)')
        
        print(f'Number of batches: {self.num_batches}')

        #
        if self.do_standard_scaling:
            if self.in_training_mode:
                means = []
                counts = []
                stddevs = []
                for p in self.data:
                    means.append(p.mean())
                    counts.append(len(p))
                    stddevs.append(p.std())
                self.mean = sum([m * c for m, c in zip(means, counts)])
                self.std  = sum([s * c for s, c in zip(stddevs, counts)])
                self.mean /= sum(counts)
                self.std  /= sum(counts)
                #
                array = numpy.array([self.mean, self.std])
                numpy.save(f'stats/raw_{self.patient_id}.npy', array)
                del means
                del counts
                del stddevs
            else:
                array = numpy.load(f'stats/raw_{self.patient_id}.npy')
                self.mean = array[0]
                self.std = array[1]
            #
            del array
        #


    def __len__(self):
        '''
        Returns the number of batches available in the current object.

        :param self: Reference to the current object.

        :return: The number of batches available in the current object.
        '''
        return self.num_batches


    def shuffle_data(self):
        '''
        Shuffles the data.

        :param self: Reference to the current object.
        '''
        #
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
        if self.balance_batches:
            half_batch = self.batch_size // 2

            # Class interictal - 0
            for sample in range(half_batch):
                idx = (batch_index * self.batch_size + sample) % len(self.indices)
                
                file_index, index, label = self.indices[idx]
                data = self.data[file_index]

                X.append(data[index : index + self.window_length])
                Y.append(label)

            # Class ictal - 1
            for sample in range(half_batch):
                idx = (batch_index * self.batch_size + sample) % len(self.indices_ictal)
                
                file_index, index, label = self.indices_ictal[idx]
                data = self.data[file_index]

                X.append(data[index : index + self.window_length])
                Y.append(label)

        else:
            for sample in range(self.batch_size):
                idx = (batch_index * self.batch_size + sample)
                
                # Do not fill last batch with samples from the beginning
                if idx < len(self.indices):
                    file_index, index, label = self.indices[idx]
                    data = self.data[file_index]

                    X.append(data[index : index + self.window_length])
                    Y.append(label)
        #

        X = numpy.array(X)
        Y = numpy.array(Y)
        #
        if self.do_standard_scaling:
            X = (X - self.mean) / self.std
        #
        if X.min() < -20. or X.max() > 20.:
            X = numpy.minimum(X,  20.)
            X = numpy.maximum(X, -20.)
        #

        X = numpy.reshape(X, (X.shape[0], 1, )  + X.shape[1:])
        Y = numpy.reshape(Y, (Y.shape[0], 1))

        return (X, Y)


#-------------------------------------------------------------------------------



if __name__=='__main__':

    """
    dg = RawRecurrentDataGenerator(index_filenames=['../indexes_detection/chb01/test.txt'],
                          window_length = 1, # in seconds
                          shift = 0.5, # in seconds
                          timesteps = 19, # in seconds
                          sampling_rate = 256, # in Hz
                          batch_size = 10,
                          do_standard_scaling = True,
                          in_training_mode = True,
                          balance_batches = True,
                          patient_id='chb01')

    for i in tqdm(range(len(dg))):
        x, y = dg[i]
        print(x.shape, y.shape)
        #print(y)
    """

    dg = RawDataGenerator2(index_filenames=['../indexes_detection/chb01/test.txt'],
                 window_length = 10, # in seconds
                 shift = 0.25, # in seconds
                 sampling_rate = 256, # in Hz
                 batch_size=20,
                 do_standard_scaling=True,
                 in_training_mode=False,
                 balance_batches=False,
                 patient_id='chb01')

    for i in tqdm(range(len(dg))):
        x, y = dg[i]
        print(x.shape, y.shape)
        #print(y)