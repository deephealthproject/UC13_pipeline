import sys
import numpy


from data_utils import decompress_pickle

'''
an example to run this script:

    find ../clean_signals/chb01/ -type f -name "*.fbank.pbz2" | python python/statistics.py

'''

file = sys.stdin

X = list()
m = s = c = 0

for line in file:
    input_filename = line.strip()

    print('processing ... ', input_filename, flush = True, end = ' ')

    x = decompress_pickle(input_filename)
    print(x.shape, x.mean(), x.std())
    x = x.flatten()
    X.append(x)
    m += x.sum()
    s += (x**2).sum()
    c += len(x)

m /= c
s /= c
s = numpy.sqrt(s - m**2)
X = numpy.hstack(X)
print(X.shape)
print(X.mean(), X.std())
print(m, s)
