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

for line in sys.stdin:
    txt_filename = line.strip()

    dir_name = os.path.dirname(txt_filename)

    current_filename = None
    current_starting_time = None
    current_ending_time = None
    labels_current_file = None
    lapse_current_file = None
    num_seizures = 0

    current_time_offset = datetime.timedelta(seconds = 0)

    subsampling_period = datetime.timedelta(seconds = 5)

    f = open(txt_filename, 'r')
    for l in f:
        l = l.strip()
        elements = l.split()

        if l.startswith('File Name:'):

            if current_filename is not None:
                print(current_filename, current_starting_time, current_ending_time, num_seizures, end = ' | ')
                #print("".join("{:d}".format(x) for x in labels_current_file))
                print()

                s = current_filename.replace('.edf', '.labels.npy')
                npy_filename = dir_name + '/' + s
                numpy.save(npy_filename, labels_current_file)
                s = current_filename.replace('.edf', '.timestamp.npy')
                npy_filename = dir_name + '/' + s
                numpy.save(npy_filename, timestamps_current_file)

            current_filename = elements[2]
            current_starting_time = None
            #current_ending_time = None
            labels_current_file = None
            lapse_current_file = None
            num_seizures = 0

        elif l.startswith('File Start Time:'):

            current_starting_time = datetime.datetime.strptime(elements[3], '%H:%M:%S') + current_time_offset

            if current_ending_time is not None:
                if current_starting_time < current_ending_time: # this ending time is the one of the previous file
                    current_time_offset += datetime.timedelta(hours = 24)
                    current_starting_time += datetime.timedelta(hours = 24)

                delta = current_starting_time - current_ending_time
                print('               delta:', delta)

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
                s = '02:' + s[3:]
                current_time_offset += datetime.timedelta(hours = 24)

            current_ending_time = datetime.datetime.strptime(s, '%H:%M:%S') + current_time_offset
                
            if current_ending_time < current_starting_time:
                raise Exception(f'{current_ending_time} is previous to {current_starting_time}')
            lapse_current_file = current_ending_time - current_starting_time
            labels_current_file = numpy.zeros((lapse_current_file - subsampling_period * 2) // subsampling_period, dtype = int)
            timestamps_current_file = [current_starting_time + (i + 1) * subsampling_period for i in range(len(labels_current_file))]

        elif l.startswith('Number of Seizures in File:'):

            num_seizures = int(elements[5])

        elif l.startswith('Seizure Start Time:'):

            s = int(elements[3])
            t = 1 + s // int(subsampling_period.total_seconds())
            if labels_current_file is None:
                print(current_filename)
            labels_current_file[t:] = 1

        elif l.startswith('Seizure End Time:'):

            s = int(elements[3])
            t = 1 + s // int(subsampling_period.total_seconds())
            labels_current_file[t:] = 0

    f.close()

    if current_filename is not None:
        print(current_filename, current_starting_time, current_ending_time, num_seizures, end = ' | ')
        #print("".join("{:d}".format(x) for x in labels_current_file))
        print()
        s = current_filename.replace('.edf', '.labels.npy')
        npy_filename = dir_name + '/' + s
        numpy.save(npy_filename, labels_current_file)
        s = current_filename.replace('.edf', '.timestamp.npy')
        npy_filename = dir_name + '/' + s
        numpy.save(npy_filename, timestamps_current_file)
