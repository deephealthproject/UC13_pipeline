import datetime
import numpy

import Preprocessor as preprocessor
import FilterBank as filter_bank
import CepstralCoefficients as cc

from matplotlib import pyplot
import scipy
from data_utils import load_file

from pyeddl import eddl
from pyeddl.tensor import Tensor
from data_utils_eeg import DataGenerator
from models_01 import model_classifier_1a, model_classifier_2a


#filename = '../clean_signals/chb01/chb01_03.edf.pbz2'
#filename = '../clean_signals/chb01/chb03_01.edf.pbz2'

filename = '../clean_signals/chb03/chb03_01'
model_id = '2a'
model_filename = 'models/model_classifier_2a-19.eddl'

with open('etc/index_view_eeg.txt', 'w') as f:
    f.write(filename)
    f.close()

dg = DataGenerator(['etc/index_view_eeg.txt'], batch_size = 1799, verbose = 1)
x, y, t = dg[0]
input_shape = (1,) + x.shape[1:]
if model_id == '1a':
    net = model_classifier_1a(input_shape, num_classes = 2, filename = model_filename)
elif model_id == '2a':
    net = model_classifier_2a(input_shape, num_classes = 2, filename = model_filename)

Y_pred = list()
for j in range(len(dg)):
    x, y_true, t = dg[j]
    x = Tensor.fromarray(x)
    (y_pred, ) = eddl.predict(net, [x])
    y_pred = y_pred.getdata()
    Y_pred = Y_pred + y_pred[:,1].tolist()

data_pieces = load_file(filename + '.edf.pbz2',
                        exclude_seizures = False,
                        do_preemphasis = False,
                        separate_seizures = False,
                        verbose = 0)

signals = data_pieces[0][0]

n_channels = signals.shape[1]

print(len(signals), signals.shape)

#preprocessors = [preprocessor.Preprocessor(256, 20000, 60000) for _ in range(n_channels)]
preprocessors = [preprocessor.Preprocessor( sampling_rate = 256, # in Hz
                                            subsampling_period = 2000, # in ms
                                            window_length = 4000, # in ms
                                            #fb_length = 20, # number of filters
                                            use_mel_scale = False,
                                            use_eeg_filtering = False,
                                            max_freq_for_filters = 70)
                            for _ in range(n_channels)]


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
    #obj = MySignal(signals[:256 * 300, ch])
    obj = preprocessor.MySignalStats(signals[:, ch])
    # a value of 0.95 for Preemphasis is used to pre-process audio signals in
    # speech recognizers, however, in the case of EEG we are interested in low
    # frequencies starting at 0.5 Hz, so using a value of 0.5 for Preemphasis
    # helps to remove the DC component while keeping the energy at low frequencies
    preprocessors[ch].preemphasis_alpha = 0.50
    obj.data += 100
    preemphasis, spectrogram, fb, fb_choi, mfcc = preprocessors[ch].preprocess_an_utterance(obj, verbose = 0)

    '''
    ed = decomposition(signal = obj.data,
                        n_components = 7,
                        alpha = 0.0,
                        window_length = 512,
                        mode = 'moving_average')
                        #mode = 'surroundings_average')
    '''

    fig, axes = pyplot.subplots(nrows = 4, ncols = 1, figsize = (9, 7))
    axis = axes[0]
    axis.grid()
    T = len(obj.data)
    axis.plot(time_axis_1[:T], obj.data, label = 'original')
    axis.plot(time_axis_1[:T], preemphasis, label = 'filtered')
    axis.set_xlabel('time')
    axis.set_ylabel('energy')
    axis.legend()
    axis.set_title(f'Channel {ch}')
    #pyplot.show()
    #
    #fig, axes = pyplot.subplots(nrows = 1, ncols = 1, figsize = (11, 5))
    axis = axes[1]
    '''
    num_channels_fft = spectrogram.shape[1]
    max_freq_for_filters = 80 # Hz
    sampling_rate = 256 # Hz
    max_ch = (num_channels_fft * max_freq_for_filters * 2) // sampling_rate
    num_filter_channels = 20
    num_band_pass_channels = max_ch // num_filter_channels
    overlapping = 2
    sp_1 = numpy.zeros([len(spectrogram), num_filter_channels])
    for k in range(num_filter_channels):
        ch_low = k * num_band_pass_channels
        ch_high = ch_low + num_band_pass_channels + overlapping
        #sp_1[:, k] = spectrogram[:, ch_low:ch_high].mean(axis = 1)
        sp_1[:, k] = spectrogram[:, k]
    #axis.pcolor(spectrogram.T)
    #axis.pcolor(sp_1.T)
    '''
    axis.pcolor(fb.T, cmap='plasma')
    #
    axis = axes[2]
    axis.grid()
    #time_axis_2 = range(len(mfcc[:, 0]))
    #axis.plot(time_axis_2, mfcc[:, 0], label = 'energy')
    #axis.plot(time_axis_2, mfcc[:, 1], label = 'CC1')
    time_axis_2 = range(len(obj.time_domain_statistics))
    axis.plot(time_axis_2, obj.time_domain_statistics[:, 0], label = 'mean')
    axis.plot(time_axis_2, obj.time_domain_statistics[:, 1], label = 'std')
    axis.plot(time_axis_2, obj.time_domain_statistics[:, 2], label = 'kurtosis')
    axis.plot(time_axis_2, obj.time_domain_statistics[:, 3], label = 'skewness')
    '''
    i = 0
    for e in ed:
        time_axis_2 = range(len(e[0]))
        axis.plot(time_axis_2, e[0], label = f'm{i}')
        i += 1
    time_axis_2 = range(len(ed[-1][1]))
    axis.plot(time_axis_2, ed[-1][1], label = f's{i}')
    '''
    axis.legend()

    axis = axes[3]
    axis.grid()
    axis.plot(range(len(Y_pred)), Y_pred, label='Class 1 probability')
    axis.legend()

    pyplot.show()
    #pyplot.savefig('eeg_view_chb1_19.png')
    #break
