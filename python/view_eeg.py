import datetime
import numpy

import Preprocessor as preprocessor
import FilterBank as filter_bank
import CepstralCoefficients as cc

from matplotlib import pyplot

from data_utils import load_file



filename = '../clean_signals/chb01/chb01_03.edf.pkl.pbz2'

data_pieces = load_file(filename,
                        exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = False,
                        verbose = 0)

signals = data_pieces[0][0]

n_channels = signals.shape[1]

print(len(signals), signals.shape)

preprocessors = [preprocessor.Preprocessor(256, 500, 1000) for _ in range(n_channels)]

class MySignal:
    def __init__(self, x):
        self.data = x
        self.n_samples = len(x)

time_axis_1 = list()
minutes = seconds = ms = 0
delta = 1000000 / 256
for i in range(signals.shape[0]):
    time_axis_1.append(datetime.datetime(year = 2021, month = 1, day = 1, hour = 0, minute = minutes, second = seconds, microsecond = int(ms)))
    ms += delta
    if ms >= 1000000:
        ms = ms - 1000000
        seconds += 1
        if seconds >= 60:
            seconds -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60
time_axis_1 = numpy.array(time_axis_1)

time_axis_2 = list()
minutes = seconds = ms = 0
for i in range(signals.shape[0] // (256//2)):
    time_axis_2.append(datetime.datetime(year = 2021, month = 1, day = 1, hour = 0, minute = minutes, second = seconds, microsecond = int(ms)))
    ms += 500000
    if ms >= 1000000:
        ms = ms - 1000000
        seconds += 1
        if seconds >= 60:
            seconds -= 60
            minutes += 1
            if minutes >= 60:
                minutes -= 60

for ch in range(n_channels):
    print(ch)
    obj = MySignal(signals[:, ch])
    preprocessors[ch].preemphasis_alfa = 0.5
    preemphasis, spectrogram, fb, fb_choi, mfcc = preprocessors[ch].preprocess_an_utterance(obj, verbose = 1)

    fig, axes = pyplot.subplots(nrows = 3, ncols = 1, figsize = (9, 7))
    axis = axes[0]
    axis.grid()
    axis.plot(time_axis_1, obj.data, label = 'original')
    axis.plot(time_axis_1, preemphasis, label = 'filtered')
    axis.set_xlabel('time')
    axis.set_ylabel('energy')
    axis.legend()
    #pyplot.show()
    #
    #fig, axes = pyplot.subplots(nrows = 1, ncols = 1, figsize = (11, 5))
    axis = axes[1]
    axis.pcolor(spectrogram.T)
    #
    axis = axes[2]
    time_axis_2 = range(len(mfcc[:, 0]))
    axis.plot(time_axis_2, mfcc[:, 0], label = 'energy')
    axis.plot(time_axis_2, mfcc[:, 1], label = 'CC1')
    pyplot.show()

