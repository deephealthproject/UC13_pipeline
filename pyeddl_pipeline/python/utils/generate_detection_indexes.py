import os
import sys
from tqdm import tqdm
from file_utils import load_file

#-------------------------------------------------------------------------------


def generate_splits(index_filename, patient_id):
    """
    Reads an index file and splits all of the files into training, validation
    and test splits, depending on the number of seizures on the files.
    It is expected to use filenames of a single patient.

    Parameters
    ----------

    :param index_filename:
        Name of the index file to use.

    :param patient_id:
        Patient id, i.e. 'chb01'

    Returns:
        None
    """
    #
    print(f'Processing patient {patient_id}')

    filenames = list()
    
    with open(index_filename, 'r') as f:
        for l in f:
            if l[0] != '#':
                filenames.append(l.strip())
        f.close()
    #
    num_files = len(filenames)
    num_seizures = 0

    ictal_len = 0
    interictal_len = 0
    
    interictal_files = list()
    ictal_files = list()

    print("Loading EDF signals...")
    for i in tqdm(range(len(filenames))):
        d_p = load_file(filenames[i] + '.edf.pbz2',
                        exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = True,
                        verbose = 0)

        ictal_file = False
        for p, label in d_p:

            if label == 0:
                interictal_len += len(p)

            elif label == 1:
                ictal_len += len(p)
                ictal_file = True
                num_seizures += 1

            else:
                raise Exception(f'Unexpected label encountered: {label} \nExpected labels are 0 and 1\n')
        #
        if ictal_file:
            ictal_files.append(i)
        else:
            interictal_files.append(i)
    #

    #if num_seizures < 7:
    #    raise Exception(f'Not enough seizures on this patient ({num_seizures}). We suggest you not to use it.')

    print(f'{num_files} records loaded!')
    print(f'Total interictal hours: {(interictal_len / 256 / 3600):2f} ')
    print(f'Total ictal hours: {(ictal_len / 256 / 3600):2f} ')
    print(f'Non-seizure recordings: {len(interictal_files)}')
    print(f'Seizure recordings: {len(ictal_files)}')

    os.makedirs('indexes_detection', exist_ok=True)
    os.makedirs(f'indexes_detection/{patient_id}', exist_ok=True)

    train_index = open(f'indexes_detection/{patient_id}/train.txt', 'w')
    val_index = open(f'indexes_detection/{patient_id}/validation.txt', 'w')
    test_index = open(f'indexes_detection/{patient_id}/test.txt', 'w')

    train_idx = int(len(interictal_files) * 0.65) # 65% training
    val_idx = int(len(interictal_files) * 0.15) # 15% validation, 20% test

    for i in interictal_files[: train_idx]:
        train_index.write(f'{filenames[i]}\n')
    
    for i in interictal_files[train_idx : train_idx+val_idx]:
        val_index.write(f'{filenames[i]}\n')
    
    for i in interictal_files[train_idx+val_idx :]:
        test_index.write(f'{filenames[i]}\n')

    train_idx = int(len(ictal_files) * 0.65) # 65% training
    val_idx = int(len(ictal_files) * 0.15) # 15% validation, 20% test
    if val_idx == 0:
        val_idx = 1 # At least 1 file

    for i in ictal_files[: train_idx]:
        train_index.write(f'{filenames[i]}\n')
    
    for i in ictal_files[train_idx : train_idx+val_idx]:
        val_index.write(f'{filenames[i]}\n')
    
    for i in ictal_files[train_idx+val_idx :]:
        test_index.write(f'{filenames[i]}\n')
    

#-------------------------------------------------------------------------------

if __name__ == '__main__':

    filename = None
    patient_id = None

    for i in range(len(sys.argv)):
        if sys.argv[i] == '--index':
            filename = sys.argv[i + 1]
        if sys.argv[i] == '--id':
            patient_id = sys.argv[i + 1]

    if filename is None or patient_id is None:
        raise Exception('You have to specify an index filename and a patient id')

    generate_splits(filename, patient_id)