import sys
import os
import numpy
from tqdm import tqdm
from data_utils import RawDataProcessor, load_file
import Preprocessor as preprocessor
from multiprocessing import Pool, cpu_count

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

        self.interictal_preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                subsampling_period = self.shift // 256 * 1000, # in ms, self.shift is provided in seconds
                                                window_length = self.window_length // 256 * 1000, # in ms
                                                fb_length = 20, # number of filters
                                                use_mel_scale = False,
                                                use_eeg_filtering = True,
                                                max_freq_for_filters = 70)
                    for _ in range(n_channels)]

        self.preictal_preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                                # Oversampling on preictal data, self.shift is provided in seconds
                                                subsampling_period = self.shift // 256 * 200, # in ms
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

        print(f'Interictal array shape: {X.shape}')
        numpy.save(dataset_dir + '/' + self.patient_id + '_interictal.npy', X)

        # Free some memory
        del self.interictal_data


        #############################################
        # Preictal signal
        #############################################
        print('Processing preictal data...')

        num_preictal_samples = 0

        for index in tqdm(range(len(self.preictal_data))):
            
            preictal_period = self.preictal_data[index]
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
        print(f'Preictal num samples: {num_preictal_samples}')
        

    def process_channel(self, ch, interictal_period):
        mss = preprocessor.MySignalStats(interictal_period, window_length = self.window_length, subsampling_period = self.shift)
        self.interictal_preprocessors[ch].preemphasis_alpha = 0.50
        preemphasis, spectrogram, fb, fb_choi, mfcc = self.interictal_preprocessors[ch].preprocess_an_utterance(mss, verbose = 0)

        assert len(fb) == len(mss.time_domain_statistics), f'{len(fb)}, {len(mss.time_domain_statistics)}'

        return (ch, fb, mss.time_domain_statistics)

# ---------------------------------------------------------------------------

if __name__ == '__main__':

    index_filenames = list()
    patient_id = 'no_patient_id'
    #
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i + 1])
        if sys.argv[i] == '--id':
            patient_id = sys.argv[i + 1]
    #
    if index_filenames is None  or  len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')
    #

    processor = RawDataProcessor(index_filenames,
                                min_interictal_length = 256 * 3600 * 4, # 4 hours
                                preictal_length = 256 * 3600, # 1 hour
                                do_standard_scaling = True,
                                do_preemphasis = False,
                                exclude_seizures = False,
                                patient_id = patient_id
                                )

    #processor = EEGDataProcessor(index_filenames, window_length = 256 * 10, shift = 256 * 10,
    #            min_interictal_length = 256 * 3600 * 4, # Select interictal samples with at least 4h of interictal period
    #            preictal_length = 256 * 3600, # 1 hour before the seizure
    #            do_preemphasis = False,
    #            exclude_seizures = False,
    #            patient_id = patient_id)
    
