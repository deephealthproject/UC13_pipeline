import sys
import numpy as np

class FilterBank:
    """
    Class for creating objects for computing the Filter Bank given the PSD
    representation as output of the FFT
    Initially designed for speech processing, but this code has been adapted
    to work with other signals.
    """

    def __init__(self, sampling_rate = 16000.0,
                 num_channels_fft = 256,
                 fb_length = None,
                 use_mel_scale = True,
                 use_eeg_filtering = False,
                 max_freq_for_filters = None):
        self.sampling_rate = sampling_rate
        self.num_channels_fft = num_channels_fft # Number of channels in the output of the FFT
        self.fb_length = fb_length
        self.use_mel_scale = use_mel_scale
        self.use_eeg_filtering = use_eeg_filtering
        self.max_freq_for_filters = max_freq_for_filters

        if self.use_mel_scale and self.use_eeg_filtering:
            raise Exception('Choose either Mel scale or EEG filtering or neither, but not both!')

        if self.use_eeg_filtering:
            self.fb_length = 8
        elif self.fb_length is None:
            # if not specified, then decide according to sampling rate
            if sampling_rate <= 8000.0:
                self.fb_length = 31
            elif sampling_rate <= 11025.0:
                self.fb_length = 31
            elif sampling_rate <= 16000.0:
                self.fb_length = 40
            elif sampling_rate <= 22050.0:
                self.fb_length = 40
            elif sampling_rate <= 44100.0:
                self.fb_length = 40
            else:
                sys.exit(1)

        if self.max_freq_for_filters is None:
            self.max_freq_for_filters = sampling_rate // 2

        self.coeffs = np.zeros([self.fb_length, self.num_channels_fft])
        self.freqs = np.linspace(0, self.sampling_rate, 1 + self.num_channels_fft)

        if self.use_mel_scale:

            min_mel = self.Hz_to_Mel(70.0) # 64.0
            max_mel = self.Hz_to_Mel(min(0.48 * sampling_rate, 0.48 * 16000.0))
            mel_incr = (max_mel - min_mel) / (self.fb_length - 1)

            current_mel = min_mel

            for i in range(self.fb_length):
                previous_freq = self.Mel_to_Hz(current_mel - mel_incr)
                current_freq  = self.Mel_to_Hz(current_mel)
                next_freq     = self.Mel_to_Hz(current_mel + mel_incr)
                sum_coeffs = 0.0
                max_coeffs = 0.0
                for k in range(len(self.freqs)- 1):
                    if self.freqs[k + 1] < previous_freq or self.freqs[k] > next_freq:
                        coeff = 0.0
                    else:
                        hz = (self.freqs[k] + self.freqs[k+1]) / 2

                        if hz <= current_freq:
                            coeff = (hz - previous_freq) / (current_freq - previous_freq)
                        else:
                            coeff = (next_freq - hz) / (next_freq - current_freq)

                    self.coeffs[i, k] = coeff
                    sum_coeffs += coeff
                    max_coeffs = max(max_coeffs, coeff)

                """
                    Should be normalized, but you can comment next loop if you
                    want to plot the coefficients
                for k in range(len(self.freqs)):
                    self.coeffs[i, k] /= sum_coeffs
                """
                self.coeffs[i, :] /= (1.0e-6 + sum_coeffs)

                current_mel = current_mel + mel_incr

        elif self.use_eeg_filtering: # use the filtering proposed in literature
            '''
            Title: Seizure Prediction in Scalp EEG Using 3D Convolutional Neural
                   Networks With an Image-Based Approach
            Authors: Ahmet Remzi Ozcan and Sarp Erturk, Senior Member, IEEE
            Journal: IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION
                     ENGINEERING, VOL. 27, NO. 11, NOVEMBER 2019


                 filter channel | spectral band
                ----------------|------------------------------------
                        1       | $\delta$     0.5 Hz -   4.0 Hz
                        2       | $\theta$     4.0 Hz -   8.0 Hz
                        3       | $\alpha$     8.0 Hz -  13.0 Hz
                        4       | $\beta$     13.0 Hz -  30.0 Hz
                        5       | $\gamma 1$  30.0 Hz -  50.0 Hz
                        6       | $\gamma 2$  50.0 Hz -  75.0 Hz
                        7       | $\gamma 3$  75.0 Hz - 100.0 Hz
                        8       | $\gamma 4$ 100.0 Hz - 128.0 Hz
                                |
            '''
            boundaries = [0.5, 4, 8, 13, 30, 50, 75, 100, 128]

            for i in range(self.fb_length):
                sum_coeffs = 0.0
                max_coeffs = 0.0
                for k in range(len(self.freqs)- 1):
                    if boundaries[i] <= self.freqs[k] and self.freqs[k + 1] <= boundaries[i + 1]:
                        coeff = 1.0
                    elif self.freqs[k] <  boundaries[i]     <= self.freqs[k + 1] or \
                         self.freqs[k] <= boundaries[i + 1] <  self.freqs[k + 1]:
                        coeff = 0.5
                    else:
                        coeff = 0.0

                    self.coeffs[i, k] = coeff
                    sum_coeffs += coeff
                    max_coeffs = max(max_coeffs, coeff)

                self.coeffs[i, :] /= (1.0e-6 + sum_coeffs)

        else: # otherwise a equalized scale is used

            delta = self.max_freq_for_filters / self.fb_length
            current_freq = 0
            for i in range(self.fb_length):
                previous_freq = current_freq - delta
                next_freq     = current_freq + delta

                sum_coeffs = 0.0
                max_coeffs = 0.0
                for k in range(len(self.freqs)- 1):
                    if self.freqs[k + 1] < previous_freq or self.freqs[k] > next_freq:
                        coeff = 0.0
                    else:
                        hz = (self.freqs[k] + self.freqs[k+1]) / 2

                        if hz <= current_freq:
                            coeff = (hz - previous_freq) / (current_freq - previous_freq)
                        else:
                            coeff = (next_freq - hz) / (next_freq - current_freq)

                    self.coeffs[i, k] = coeff
                    sum_coeffs += coeff
                    max_coeffs = max(max_coeffs, coeff)

                self.coeffs[i, :] /= (1.0e-6 + sum_coeffs)
                current_freq += delta
            # end for

        # Attributes for implementing the Choi spectral flooring
        self.alpha = np.zeros(self.fb_length)
        self.beta = np.zeros(self.fb_length)
        self.gamma = np.zeros(self.fb_length)
        self.noise_threshold = np.ones(self.fb_length)
        self.alpha_of_noise_threshold = 0.1

        for i in range(self.fb_length):
            self.beta[i] = 1e-3
            self.gamma[i] = 0.4
            self.noise_threshold[i] = 1.0

        #self.normalize_using_the_maximum = False
        #self.apply_Choi_spectral_flooring = False

    def get_num_filters(self):
        return self.fb_length

    def Hz_to_Mel(self, Hz):
        #return 1127.0 * np.log1p( Hz / 700 )
        return 2595.0 * np.log10(1 + Hz / 700)

    def Mel_to_Hz(self, mels):
        #return 700 * (np.exp(mels / 1127.0 ) - 1)
        return 700 * (10.0 ** (mels / 2595.0) - 1)

    def centralBin(self, freq):
        i = int(round((2. * self.num_channels_fft * freq) / self.sampling_rate))
        i = max(i, 0)
        i = min(i, self.num_channels_fft - 1)
        return i

    """
        Computes the filter bank given the power spectrum as outcome of the FFT
    """
    def compute_filter_bank(self, psd):
        fb = np.dot(self.coeffs, psd)
        for i in range(len(fb)):
            fb[i] = np.log10(1.0 + fb[i])
        return fb

    """
        Computes the filter bank given the power spectrum as outcome of the FFT.
        This method also obtains the filtered version of the filter bank
        following the Choi approach
    """
    def compute_filter_bank(self, psd, fb_log, choi = None, is_silence = False):
        fb_no_log = np.dot(self.coeffs, psd)
        sum_alpha = 1.0 # To avoid division by zero
        max_alpha = 1.0 # Idem
        for i in range(len(fb_no_log)):
            self.alpha[i] = np.log10(1 + fb_no_log[i] / self.noise_threshold[i])
            sum_alpha += self.alpha[i]
            max_alpha = max(self.alpha[i], max_alpha)
            fb_log[i] = np.log10(1.0 + fb_no_log[i])
        if choi is not None:
            for i in range(len(fb_no_log)):
                self.alpha[i] /= sum_alpha
                #self.alpha[i] /= max_alpha
                choi[i] = self.alpha[i] * np.log1p(self.beta[i] * max(fb_no_log[i] - self.noise_threshold[i], self.gamma[i] * fb_no_log[i]))
                #choi[i] = min( 0.2, choi[i] )

            if is_silence :
                self.update_noise_threshold(fb_no_log)


    def update_noise_threshold(self, fb):
        for i in range(len(fb)):
            self.noise_threshold[i] += self.alpha_of_noise_threshold * (fb[i] - self.noise_threshold[i])
            self.noise_threshold[i] = max(self.noise_threshold[i], 1.0)


    """ Call this function as fb.view_filters() where fb is an object of this class. """
    def view_filters(self):

        import matplotlib.pyplot as plt

        for i in range(self.coeffs.shape[0]):
            plt.plot(self.coeffs[i])
        plt.grid()
        plt.show()


#FilterBank().view_filters()
