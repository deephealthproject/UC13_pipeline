import os
import sys
import math
import numpy as np
from scipy import signal, fftpack, constants, stats

from FilterBank import FilterBank
from CepstralCoefficients import CepstralCoefficients


class Preprocessor:
    """
        Class for containing some functions related for performing the
        parametrization of the speech signal
    """

    def __init__(self,  sampling_rate,
                        subsampling_period,
                        window_length,
                        fb_length = None,
                        max_freq_for_filters = None,
                        use_mel_scale = True,
                        use_eeg_filtering = False):
        ####################################
        self.sampling_rate = sampling_rate # In Hertz
        self.subsampling_period = subsampling_period # In milliseconds
        self.window_length = window_length # In milliseconds
        ####################################
        self.dc_removal_K = (1.0 - 1.0 / 1024.0)
        self.dc_removal_x_ant = 0.0
        self.dc_removal_y_ant = 0.0
        ####################################
        self.preemphasis_alpha = 0.95
        self.preemphasis_x_ant = 0.0
        ####################################
        self.hamming_window = signal.hamming((self.sampling_rate * self.window_length) // 1000, True)
        self.len_fft_window = 2
        while self.len_fft_window < len(self.hamming_window):
            self.len_fft_window *= 2
        self.filter_bank = FilterBank(self.sampling_rate,
                                        self.len_fft_window // 2,
                                        fb_length = fb_length,
                                        use_mel_scale = use_mel_scale,
                                        use_eeg_filtering = use_eeg_filtering,
                                        max_freq_for_filters = max_freq_for_filters)
        ####################################
        self.cepstral = CepstralCoefficients(self.filter_bank.get_num_filters(), 13)
        ####################################
        #self.log_silence_threshold = np.log10( self.hamming_window.sum() * 8**2 )
        self.log_silence_threshold = 4.0



    def dc_removal(self, input_signal, output_signal):

        for t in range(len(input_signal)):
            temp = input_signal[t] # 'temp' is used to prevent error in the case 'input_signal' and 'output_signal' are the same array
            #print(t, input_signal[t], self.dc_removal_x_ant, self.dc_removal_y_ant, self.dc_removal_K)
            output_signal[t] = (1. * input_signal[t] - self.dc_removal_x_ant) + self.dc_removal_y_ant * self.dc_removal_K
            self.dc_removal_x_ant = temp
            self.dc_removal_y_ant = output_signal[t]
        ### END OF dc_removal() ###

    def preemphasis(self, input_signal, output_signal):

        for t in range(len(input_signal)):
            temp = input_signal[t] # 'temp' is used to prevent error in the case 'input_signal' and 'output_signal' are the same array
            output_signal[t] = input_signal[t] - self.preemphasis_alpha * self.preemphasis_x_ant
            self.preemphasis_x_ant = temp
        ### END OF preemphasis() ###

    """
        It is assumed that X is a piece of wave signal obtained after applying
        the DC removal filter, the preemphasis filter and the hamming window
    """
    def extract_features(self, X, initial_frames):
        ### Computing the energy for an acoustic frame
        logE = 10.0 * (np.log10(1. + np.dot(X, X)) - self.log_silence_threshold)
        normE = 2.5 * min(50.0, max(0.0, logE)) / 50 - 1.5
        ### Computing the FFT then the PSD
        fft = fftpack.rfft(X, self.len_fft_window)
        psd = np.zeros(self.len_fft_window // 2)
        psd[0] = np.abs(fft[0])
        for c in range(2, len(fft), 2):
            psd[c // 2] = np.sqrt(fft[c - 1] * fft[c - 1] + fft[c] * fft[c])
        ### Computing the filterbank
        non_filtered = np.zeros(self.filter_bank.get_num_filters())
        choi = np.zeros(self.filter_bank.get_num_filters())
        self.filter_bank.compute_filter_bank(psd, non_filtered, choi, (normE <= -1.0) or initial_frames)
        ### Applying logarithm to the PSD after being used for computing the filterbank
        for c in range(len(psd)):
            psd[c] = 10 * np.log10(1. + psd[c])
        ### Compute the cepstral coefficients
        #cc = self.cepstral.compute( non_filtered ) # Decide which to use, this or the filtered
        cc = self.cepstral.compute(choi)
        self.cepstral.liftering(cc) # Comment this if you don't need to lifter the cepstral coefficients
        self.cepstral.equalization(cc, logE)
        cc[0] = normE # Comment this line if you want to use choi.sum() as the value of energy
        ###
        return psd, non_filtered, choi, cc
        ### END OF extract_features() ###


    def preprocess_an_utterance(self, utterance, verbose = 0):
        preemphasis = np.zeros(utterance.n_samples)
        self.dc_removal(utterance.data, preemphasis)
        self.preemphasis(preemphasis, preemphasis)
        ##################
        lhw = len(self.hamming_window)
        delta = int(self.subsampling_period * self.sampling_rate) // 1000
        ##################
        spectrogram = list()
        fb = list()
        fb_choi = list()
        mfcc = list()
        #for t in range(0, utterance.n_samples - lhw, delta):
        t = 0
        while t + lhw <= utterance.n_samples:
            X = preemphasis[t:t+lhw] * self.hamming_window
            psd, non_filtered, choi, cc = self.extract_features(X, (1.0 * t / self.sampling_rate <= 0.100))

            # This is redundant, the unique purpose is for debug
            ##E = 10.0 * np.log10(np.dot(X, X))
            E = 10.0 * (np.log10(1. + np.dot(X, X)) - self.log_silence_threshold)
            if verbose > 0:
                print("time: %10.3f seg    energy: %20.6f dB    cc0: %10.6f" % ( (1.0 * t / self.sampling_rate), E, cc[0]))

            spectrogram.append(psd)
            fb.append(non_filtered)
            fb_choi.append(choi)
            mfcc.append(cc)
            #
            t += delta

        spectrogram = np.array(spectrogram)
        fb = np.array(fb)
        fb_choi = np.array(fb_choi)
        mfcc = np.array(mfcc)

        return preemphasis, spectrogram, fb, fb_choi, mfcc
        ### END OF preprocess_an_utterance() ###



class MySignalStats:
    """
        Class for computing the statistics in the time domain for EEG, as
        indicated in

            Title: Seizure Prediction in Scalp EEG Using 3D Convolutional Neural
                   Networks With an Image-Based Approach
            Authors: Ahmet Remzi Ozcan and Sarp Erturk, Senior Member, IEEE
            Journal: IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION
                     ENGINEERING, VOL. 27, NO. 11, NOVEMBER 2019

    """
    def __init__(self, data, window_length = 256 * 4, subsampling_period = 256 * 2):
        self.data = data
        self.n_samples = len(self.data)

        if window_length == subsampling_period:
            self.n_subsamples = len(self.data) // subsampling_period
        else:
            self.n_subsamples = (len(self.data) - subsampling_period) // subsampling_period

        time_domain_statistics = list()
        i = window_length
        count = 0
        while i <= self.n_samples:
            x = self.data[i - window_length : i]
            mean = x.mean()
            std  = x.std()
            kurt = stats.kurtosis(x)
            skew = stats.mstats.skew(x)
            if np.isnan(kurt):
                #print('kurtosis is NaN', " ".join("{:e}".format(_) for _ in x))
                print('kurtosis is NaN', i, sum(abs(x)))
                sys.exit(1)
            if np.isnan(skew.data):
                #print('skewness is NaN', " ".join("{:e}".format(_) for _ in x))
                print('kurtosis is NaN', i, sum(abs(x)))
                sys.exit(1)

            d_x  = self.time_derivative(  x, [-3, -2, -1, 0, 1, 2, 3])
            dd_x = self.time_derivative(d_x, [-3, -2, -1, 0, 1, 2, 3])
            #
            mobility = math.sqrt(d_x.var() / (1.0e-3 + x.var()))
            d_mobility = math.sqrt(dd_x.var() / (1.0e-3 + d_x.var()))
            complexity = d_mobility / (1.0e-3 + mobility)

            time_domain_statistics.append([mean, std, kurt, skew.data, mobility, complexity])
            #
            i += subsampling_period

        self.time_domain_statistics = np.array(time_domain_statistics)
        assert len(self.time_domain_statistics) == self.n_subsamples

    def time_derivative(self, x, alphas):
        y = np.convolve(x, np.array(alphas), mode = 'valid')
        y /= sum(abs(np.array(alphas)))
        return y

    def get_mean(self):
        return self.stats[:, 0]

    def get_std(self):
        return self.stats[:, 1]

    def get_kurtosis(self):
        return self.stats[:, 2]

    def get_skewness(self):
        return self.stats[:, 3]

    def get_mobility(self):
        return self.stats[:, 4]

    def get_complexity(self):
        return self.stats[:, 5]
