
import Preprocessor as prep
import FilterBank as fb
import CepstralCoefficients as cc

import matplotlib.pyplot as plt

prep20ms = prep.Preprocessor(256, 500, 1000)
prep25ms = prep.Preprocessor(256, 500, 1000)
prep30ms = prep.Preprocessor(256, 500, 1000)

plt.plot(prep20ms.hamming_window)
plt.plot(prep25ms.hamming_window)
plt.plot(prep30ms.hamming_window)
plt.show()


#   sampling_rate * window_length (in ms) / 1000 (because ms)
x = 256 * 10000 / 1000
num_channels_fft = 1
while num_channels_fft < x:
    num_channels_fft *= 2

print(x, num_channels_fft)

fb.FilterBank(sample_rate = 256,
            num_channels_fft = int(num_channels_fft // 2),
            fb_length = 20,
            use_mel_scale = False,
            max_freq_for_filters = 70).view_filters()

cc.CepstralCoefficients(20, 13).view_dct()
