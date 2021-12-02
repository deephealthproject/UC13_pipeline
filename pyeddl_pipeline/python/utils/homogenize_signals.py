import os
import sys

import pyedflib.highlevel as hl
import multiprocessing as mp
from file_utils import compress_to_pickle


def process_metadata(summary, filename):
    """

    Get metadata from a record: signal names, number of seizures and
        start and end time of every seizure.

    :param string summary
        Path to the summary file.

    :param string filename
        Name of the file to process.

    :return dict metadata
        Dictionary with all the metadata.

    """
    f = open(summary, 'r')

    metadata = {}
    lines = f.readlines()
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2]==filename:
            j = i+1
            processed=False
            while (not processed):
                if (lines[j].split()[0]=='Number'):
                    seizures = int(lines[j].split()[-1])
                    processed=True
                j = j + 1

            # If file has seizures get start and end time
            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    # Save start and end time of each seizure
                    processed = False
                    while (not processed):
                        l = lines[j].split()
                        #print(l)

                        if l[0]=='Seizure' and 'Start' in l:
                            start = int(l[-2]) * 256 - 1                # Index of start time
                            end = int(lines[j+1].split()[-2]) * 256 - 1 # Index of end time
                            processed=True
                        j = j + 1
                    times.append((start,end))

            metadata['seizures']=seizures
            metadata['times']=times
    
    return metadata


def process_reference_file(filename):
    """

    Process the reference file and return a list of signal names.

    :param string filename
        Name of the reference file.
    
    :return list signal_reference
        List with the names of the reference file signals.

    """

    signal_reference = []
    with open(filename, 'r') as ref:
        for line in ref:
            line = line.split()
            if len(line) == 0: continue
            #
            if line[0]=='Channels' and line[1]=='Changed':
                break
            #
            if line[0]=='Channel':
                name = line[2]
                #inv_name = name.split('-')[1] + '-' + name.split('-')[0]
                #if name not in signal_reference:
                #    signal_reference.append(line[2])
                if name in signal_reference:
                    name = name + '_2'
                signal_reference.append(name)

    
    return signal_reference


def process_file(filename,  signal_reference, store_path):
    """

    Process a .edf file and stores it in store_path.
    It saves a dictionary with signal names as keys and data as values.
    It also has an extra key for metadata.

    :param string filename
        File to process.

    :param list signal_reference
        List with signal names of the reference file.

    :param string store_path
        Directory where to store the data after compressing.

    :return None

    """
    
    ignore_list = ['chb13_04.edf', 'chb12_27.edf', 'chb12_28.edf', 'chb12_29.edf', 'chb15_01.edf']

    dir_patient = os.path.dirname(filename)
    patient_id = os.path.basename(dir_patient)
    filename_b = os.path.basename(filename)
    summary_filename = dir_patient + '/' + patient_id + "-summary.txt"

    if filename_b in ignore_list:
        print('Ignoring file', filename)
    else:
        print('Processing file ', filename)

        signal_dict = {}
        signal_names = []

        metadata = process_metadata(summary_filename, filename_b)
        metadata['channels']=signal_reference
        signals, signal_headers, header = hl.read_edf(filename, digital=False)

        if len(signal_headers) < 23: # Before was len(signal_reference)
            print("Not enough signals on file ", filename)
        else:
            # Process file
            for i in range(len(signal_headers)):
                name = signal_headers[i].get('label')
                if name not in signal_names:
                    signal_names.append(name)
                else:
                    signal_names.append(name + '_2')
            
            for i in range(len(signal_names)):
                if signal_names[i] in signal_reference:
                    signal_dict[signal_names[i]]=signals[i]

            if len(signal_dict) != len(signal_reference): print("ERROR ON FILE ", filename)

            signal_dict['metadata'] = metadata

            compress_to_pickle(store_path, signal_dict)


if __name__=='__main__':

    signals_path = '../../UC13/physionet.org/files/chbmit/1.0.0' # Path to the data main directory
    clean_path = '../clean_signals' # Path where to store clean data

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--signals-path':
            signals_path = sys.argv[i+1]
        elif sys.argv[i] == '--clean-path':
            clean_path = sys.argv[i+1]

    # Reference file
    # All processed files will have same signals as the reference
    reference_file = signals_path + '/chb01/chb01-summary.txt'

    signal_reference = process_reference_file(reference_file)

    params = []
    for d in os.listdir(signals_path):
        if d.startswith('chb'):
            for f in os.listdir(signals_path + '/' + d):
                if f.endswith('.edf'):
                    if not os.path.exists(clean_path + '/' + d):
                        os.makedirs(clean_path + '/' + d)
                    # this prevents to overwrite existing EDF files, the data scientist must
                    # remove the old ones before executing this Python code to regenerate EDF files.
                    if not os.path.exists(clean_path + '/' + d + '/' + f + '.pbz2'):
                        params.append((signals_path + '/' + d + '/' + f, signal_reference, clean_path + '/' + d + '/' + f))
                    else:
                        print(clean_path + '/' + d + '/' + f, 'already exists!')
            #
        #
    #

    with mp.Pool(mp.cpu_count()) as pool:
        pool_output = pool.starmap(process_file, params)
