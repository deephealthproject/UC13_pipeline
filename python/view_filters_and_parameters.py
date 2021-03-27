
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


fb.FilterBank(sample_rate = 256, num_channels_fft = 128, fb_length = 40, mel_scale = False).view_filters()

cc.CepstralCoefficients(40, 13).view_dct()
