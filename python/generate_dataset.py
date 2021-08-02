import sys
import os
import numpy
from tqdm import tqdm
from data_utils import RawDataProcessor, EEGDataProcessor
import Preprocessor as preprocessor
from multiprocessing import Pool, cpu_count


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

    #processor = RawDataProcessor(index_filenames,
    #                            min_interictal_length = 256 * 3600 * 4, # 4 hours
    #                            preictal_length = 256 * 3600, # 1 hour
    #                            do_standard_scaling = True,
    #                            do_preemphasis = False,
    #                            exclude_seizures = False,
    #                            patient_id = patient_id
    #                            )

    processor = EEGDataProcessor(index_filenames, window_length = 256 * 10, shift = 256 * 10,
                min_interictal_length = 256 * 3600 * 4, # Select interictal samples with at least 4h of interictal period
                preictal_length = 256 * 3600, # 1 hour before the seizure
                do_preemphasis = False,
                exclude_seizures = False,
                patient_id = patient_id)
    
