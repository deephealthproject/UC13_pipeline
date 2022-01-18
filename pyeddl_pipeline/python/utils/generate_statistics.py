import os
import sys
from tqdm import tqdm
from file_utils import load_file

#-------------------------------------------------------------------------------


def generate_stats(index_filenames, patient_id):
    """
    Reads an index file of a patient and prints the number of seizures,
    the number of ictal hours and the number of interictal hours.

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
    filenames = list()
    for index_filename in index_filenames:
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


    for i in tqdm(range(len(filenames))):
        d_p = load_file(filenames[i] + '.edf.pbz2',
                        exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = True,
                        verbose = 0)

        for p, label in d_p:

            if label == 0:
                interictal_len += len(p)
            elif label == 1:
                ictal_len += len(p)
                num_seizures += 1

            else:
                raise Exception(f'Unexpected label encountered: {label} \nExpected labels are 0 and 1\n')
        #
    #

    interictal_hours = (interictal_len / 256 / 3600)
    ictal_hours = (ictal_len / 256 / 3600)
    total_hours = interictal_hours + ictal_hours
    seizures_per_hour = num_seizures / total_hours

    print('******************************************************')
    print(f'Patient {patient_id} summary')
    print(f'Number of seizures:     {num_seizures}')
    print(f'Total interictal hours: {interictal_hours:.2f}')
    print(f'Total ictal hours:      {ictal_hours:.2f} ')
    print(f'Seizures per hour:      {seizures_per_hour:.2f}')
    print()


#-------------------------------------------------------------------------------

if __name__ == '__main__':

    for patient in [3, 5, 8, 12, 14, 15, 24]:
        patient_id = 'chb' + ('0' + str(patient))[-2:]
        filenames = []
        filenames.append(f'../indexes_detection/{patient_id}/train.txt')
        filenames.append(f'../indexes_detection/{patient_id}/validation.txt')
        filenames.append(f'../indexes_detection/{patient_id}/test.txt')
        generate_stats(filenames, patient_id)
