"""
This file has functions to process edf files.
"""
import os
import sys
import bz2
import _pickle as cPickle
import numpy
from multiprocessing import Pool


# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
def compress_to_pickle(filename, obj):
    '''
    Save a Python object into a compressed pickle file
    '''
    if not filename.endswith('.pbz2'):
        filename += '.pbz2'
    with bz2.BZ2File(filename, 'w') as f:
        cPickle.dump(obj, f)


# ------------------------------------------------------------------------------
def load_data_one_patient(base_dir = '.',
                            n_processes = 8,
                            verbose = 0,
                            exclude_seizures = False,
                            do_preemphasis = False):
    '''
    Load data from one directory asuming all files belong to the same patient.

    :param str base_dir:
        Path to the directory where the files with extension ".edf.pbz2" are located.

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

    filenames = list()
    for filename in os.listdir(base_dir):
        if filename.endswith(".edf.pbz2"):
            filenames.append(base_dir + '/' + filename)

    return load_files(filenames,
                        n_processes = n_processes,
                        verbose = verbose,
                        exclude_seizures = exclude_seizures,
                        do_preemphasis = do_preemphasis)


# ------------------------------------------------------------------------------
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
            if d is not None:
                data_pieces.append(d)

    return data_pieces


# ------------------------------------------------------------------------------
def load_file_old(filename, exclude_seizures = False,
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


# ------------------------------------------------------------------------------
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
        episodes.sort(key = lambda a: a[0], reverse = False)

    if verbose > 1:
        print(signal_ids)
        print(num_seizures)
        print(episodes)

    if do_preemphasis:
        alpha = 0.98
        for key in signal_ids:
            x = signal_dict[key].copy()
            signal_dict[key][1:] = x[1:] - alpha * x[:-1]
            signal_dict[key][0] = 0

    data_pieces = list()
    if separate_seizures and episodes is not None:
        i0 = 0
        for boundaries in episodes:
            i1 = boundaries[0]

            # we are in LABEL 0 (no ICTAL periods)
            _signal = numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T
            data_pieces.append((_signal, 0))

            # we are in LABEL 1 (ICTAL periods)
            i0 = boundaries[0]
            i1 = boundaries[1]
            if not exclude_seizures:
                _signal = numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T
                data_pieces.append((_signal, 1))
            
            # Update indexes
            i0 = i1 + 1

        # Add last part of the signal
        # we are in LABEL 0 (no ICTAL periods)
        i1 = len(signal_dict[signal_ids[0]])
        _signal = numpy.array([signal_dict[signal_id][i0:i1] for signal_id in signal_ids]).T
        data_pieces.append((_signal, 0))

    else:
        # Do not separate any seizure, just return the entire signal
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


# ------------------------------------------------------------------------------
def load_file_eeg(filename, verbose = 0):
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
