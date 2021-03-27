import sys
import os
import csv
import numpy


def get_label_from_filename(filename):
    parts = filename.split(sep = '/')
    parts = parts[1].split(sep = '.')
    parts = parts[0].split(sep = '_')
    return int(parts[3])

def load_index_from_csv(filename):
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter = ';')
    index = list()
    header = next(reader) # we assume the first line is the header
    columns = [str(s) for s in header]
    for values in reader:
        d = dict()
        for col, value in zip(columns,values):
            if col in ['class']:
                d[col] = int(value)
            else:
                d[col] = value
        index.append(d)
    f.close()
    return index

def compute_derivatives(x):
    filter = numpy.array([-2, -1, 0, 1, 2])
    d1 = list()
    for j in range(x.shape[1]):
        d1.append(numpy.convolve(x[:, j], filter, mode = 'same'))
    d1 = numpy.vstack(d1).T
    d2 = list()
    for j in range(x.shape[1]):
        d2.append(numpy.convolve(d1[:, j], filter, mode = 'same'))
    d2 = numpy.vstack(d2).T
    return numpy.hstack([x, d1, d2])



def show_MFCC_statistics(mfcc):
    print("MFCC statistics:")
    for k in range(mfcc.shape[1]):
        print("mfcc %2d   mean %10.6f range %10.6f  limits %10.6f %10.6f" %
              (k, mfcc[:,k].mean(), mfcc[:,k].ptp(), mfcc[:,k].min(), mfcc[:,k].max()))
