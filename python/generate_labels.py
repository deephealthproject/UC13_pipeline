import os
import sys
import time
import datetime
import numpy


'''
File Name: chb01_03.edf
File Start Time: 13:43:04
File End Time: 14:43:04
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds
'''

index_to_label = ['inter-ictal', 'ictal', 'pre-ictal', 'post-ictal']
label_to_index = {'inter-ictal': 0, 'ictal': 1, 'pre-ictal': 2, 'post-ictal': 3}

pre_ictal_lapse = datetime.timedelta(seconds = 3600) # one hour before the seizure
post_ictal_lapse = datetime.timedelta(seconds = 1800) # half an hour after the seizure


if __name__ == '__main__':
    total_samples = 0
    total_on_seizure = [0 for x in label_to_index.keys()]

    for line in sys.stdin:
        txt_filename = line.strip()

        dir_name = os.path.dirname(txt_filename)

        current_time_offset = datetime.timedelta(seconds = 0)

        subsampling_period = datetime.timedelta(seconds = 2)

        file_sequence = list()

        f = open(txt_filename, 'r')
        for l in f:
            l = l.strip()
            elements = l.split()

            if l.startswith('File Name:'):

                file_sequence.append(dict(filename = elements[2],
                                        starting_time = None,
                                        ending_time = None,
                                        labels = None,
                                        timestamps = None,
                                        lapse = None,
                                        num_seizures = 0))

            elif l.startswith('File Start Time:'):

                file_sequence[-1]['starting_time'] = datetime.datetime.strptime(elements[3], '%H:%M:%S') + current_time_offset

            elif l.startswith('File End Time:'):

                s = elements[3]
                if s[:2] == '24':
                    s = '00:' + s[3:]
                    current_time_offset += datetime.timedelta(hours = 24)
                elif s[:2] == '25':
                    s = '01:' + s[3:]
                    current_time_offset += datetime.timedelta(hours = 24)
                elif s[:2] == '26':
                    s = '02:' + s[3:]
                    current_time_offset += datetime.timedelta(hours = 24)
                elif s[:2] == '27':
                    s = '03:' + s[3:]
                    current_time_offset += datetime.timedelta(hours = 24)

                file_sequence[-1]['ending_time'] = datetime.datetime.strptime(s, '%H:%M:%S') + current_time_offset
                    
                if file_sequence[-1]['ending_time'] < file_sequence[-1]['starting_time']:
                    raise Exception(str(file_sequence[-1]['ending_time']) + ' is previous to ' + str(file_sequence[-1]['starting_time']))
                lapse = file_sequence[-1]['ending_time'] - file_sequence[-1]['starting_time']
                file_sequence[-1]['lapse'] = lapse
                file_sequence[-1]['labels'] = labels = numpy.zeros((lapse - subsampling_period * 1) // subsampling_period, dtype = int)
                file_sequence[-1]['timestamps'] = [file_sequence[-1]['starting_time'] + (i + 1) * subsampling_period for i in range(len(labels))]
                labels = None
                lapse = None

            elif l.startswith('Number of Seizures in File:'):

                file_sequence[-1]['num_seizures'] = int(elements[5])

            elif l.startswith('Seizure'):
                if elements[1] == 'Start' and elements[2] == 'Time:':

                    s = int(elements[3])
                    l = 1

                elif elements[2] == 'Start' and elements[3] == 'Time:':

                    s = int(elements[4])
                    l = 1

                elif elements[1] == 'End' and elements[2] == 'Time:':

                    s = int(elements[3])
                    l = 0

                elif elements[2] == 'End' and elements[3] == 'Time:':

                    s = int(elements[4])
                    l = 0

                else:
                    raise Exception('unrecognized seizure line')

                t = 1 + s // int(subsampling_period.total_seconds())
                file_sequence[-1]['labels'][t:] = l

        f.close()

        '''
            First the pre-ictal periods are set
        '''
        first_ictal_timestamp = None
        for j in range(len(file_sequence) - 1, -1, -1):
            file_data = file_sequence[j]
            for i in range(len(file_data['labels']) - 1, -1, -1):
                l = file_data['labels'][i]
                t = file_data['timestamps'][i]
                if l == label_to_index['ictal']:
                    first_ictal_timestamp = t
                elif first_ictal_timestamp is not None: #elif l == label_to_index['inter-ictal']:
                    if t < first_ictal_timestamp and (first_ictal_timestamp - t) <= pre_ictal_lapse:
                        file_data['labels'][i] = label_to_index['pre-ictal']

        '''
            Second the post-ictal periods are set, because these periods have more priority than
            the pre-ictal ones
        '''
        last_ictal_timestamp = None
        for file_data in file_sequence:
            for i in range(len(file_data['labels'])):
                l = file_data['labels'][i]
                t = file_data['timestamps'][i]
                if l == label_to_index['ictal']:
                    last_ictal_timestamp = t
                elif last_ictal_timestamp is not None: # elif l == label_to_index['inter-ictal']:
                    if t > last_ictal_timestamp and (t - last_ictal_timestamp) <= post_ictal_lapse:
                        file_data['labels'][i] = label_to_index['post-ictal']


        for file_data in file_sequence:
            print(file_data['filename'], file_data['starting_time'], file_data['ending_time'], file_data['num_seizures'], end = ' | ')
            print()

            labels = file_data['labels']

            for key in label_to_index.keys():
                i = label_to_index[key]
                total_on_seizure[i] += sum(labels == i)

            total_samples    += len(labels)

            s = file_data['filename'].replace('.edf', '.labels.npy')
            npy_filename = dir_name + '/' + s
            numpy.save(npy_filename, file_data['labels'])
            s = file_data['filename'].replace('.edf', '.timestamp.npy')
            npy_filename = dir_name + '/' + s
            numpy.save(npy_filename, file_data['timestamps'])


    print()
    print('number of samples per class:')
    count = 0
    for key in label_to_index.keys():
        i = label_to_index[key]
        print(' %12d  %s' % (total_on_seizure[i], key))
        count += total_on_seizure[i]
    print(' %12d  %s' % (total_samples, 'total'))
    print()
    if count != total_samples:
        print('ERROR: differences in counting samples %d vs %s' % (count, total_samples))
