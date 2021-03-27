import numpy as np

class FilterBank:
    """
    Class for creating objects for computing the Filter Bank given the PSD
    representation as output of the FFT
    Initially designed for speech processing, but this code has been adapted
    to work with other signals.
    """

    def __init__(self, sample_rate = 16000.0, num_channels_fft = 256, fb_length = None, mel_scale = True):
        self.sample_rate = sample_rate
        self.num_channels_fft = num_channels_fft # Number of channels in the output of the FFT
        self.fb_length = fb_length
        self.mel_scale = mel_scale

        if self.fb_length is None:
            # if not specified, then decide according to sampling rate
            if sample_rate <= 8000.0:
                self.fb_length = 31
            elif sample_rate <= 11025.0:
                self.fb_length = 31
            elif sample_rate <= 16000.0:
                self.fb_length = 40
            elif sample_rate <= 22050.0:
                self.fb_length = 40
            elif sample_rate <= 44100.0:
                self.fb_length = 40
            else:
                sys.exit(1)

        self.coeffs = np.zeros([self.fb_length, self.num_channels_fft])
        self.freqs = np.zeros(self.num_channels_fft)

        if self.mel_scale:
            delta = sample_rate / (2.0 * self.num_channels_fft)
            self.freqs[0] = delta;
            for i in range(1, self.num_channels_fft):
                self.freqs[i] = self.freqs[i - 1] + delta

            #print("freqs: ", self.freqs)

            min_mel = self.Hz_to_Mel(70.0) # 64.0
            max_mel = self.Hz_to_Mel(min(0.48 * sample_rate, 0.48*16000.0))
            mel_incr = (max_mel - min_mel) / (self.fb_length - 1)

            current_mel = min_mel

            for i in range(self.fb_length):
                previous_freq = self.Mel_to_Hz(current_mel - mel_incr)
                current_freq  = self.Mel_to_Hz(current_mel)
                next_freq     = self.Mel_to_Hz(current_mel + mel_incr)
                sum_coeffs = 0.0
                max_coeffs = 0.0
                for k in range(len(self.freqs)):
                    if self.freqs[k] >= previous_freq and self.freqs[k] <= current_freq:
                        coeff = (self.freqs[k] - previous_freq) / (current_freq - previous_freq)
                    elif self.freqs[k] >= current_freq and self.freqs[k] <= next_freq:
                        coeff = (next_freq - self.freqs[k]) / (next_freq - current_freq)
                    else:
                        coeff = 0.0

                    self.coeffs[i, k] = coeff
                    sum_coeffs += coeff
                    max_coeffs = max(max_coeffs, coeff)

                """"
                    Should be normalized, but you can comment next loop if you
                    want to plot the coefficients
                """
                for k in range(len(self.freqs)):
                    self.coeffs[i, k] /= sum_coeffs

                current_mel = current_mel + mel_incr

        else: # otherwise a equalized scale is used

            self.coeffs = np.zeros([self.fb_length, self.num_channels_fft])
            self.freqs = np.zeros(self.num_channels_fft)

            delta = sample_rate / (2.0 * self.num_channels_fft)
            self.freqs[0] = delta;
            for i in range(1, self.num_channels_fft):
                self.freqs[i] = self.freqs[i - 1] + delta

            delta = self.num_channels_fft / self.fb_length
            current_freq = 0
            next_freq = delta
            for i in range(self.fb_length):
                previous_freq = current_freq
                current_freq  = current_freq + delta
                next_freq     = next_freq + delta

                sum_coeffs = 0.0
                max_coeffs = 0.0
                for k in range(len(self.freqs)):
                    if self.freqs[k] >= previous_freq and self.freqs[k] <= current_freq:
                        coeff = (self.freqs[k] - previous_freq) / (current_freq - previous_freq)
                    elif self.freqs[k] >= current_freq and self.freqs[k] <= next_freq:
                        coeff = (next_freq - self.freqs[k]) / (next_freq - current_freq)
                    else:
                        coeff = 0.0

                    self.coeffs[i, k] = coeff
                    sum_coeffs += coeff
                    max_coeffs = max(max_coeffs, coeff)


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
        i = int(round((2. * self.num_channels_fft * freq) / self.sample_rate))
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
    def view_filters( self ):

        import matplotlib.pyplot as plt

        for i in range(self.coeffs.shape[0]):
            plt.plot(self.coeffs[i])
        plt.grid()
        plt.show()


#FilterBank().view_filters()
