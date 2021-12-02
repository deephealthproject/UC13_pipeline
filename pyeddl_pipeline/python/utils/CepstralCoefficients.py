
import numpy as np
from scipy import constants

class CepstralCoefficients:

    def __init__(self, num_filters, num_coeffs):
        self.table = np.zeros([num_coeffs, num_filters])
        self.num_filters = num_filters
        self.num_coeffs = num_coeffs
        self.liftering_weights = np.ones(self.num_coeffs)
        self.liftering_window_length = 2 * (self.num_coeffs - 1)
        self.equalization_lambda = 0.0087890625
        self.equalization_bias = np.zeros(self.num_coeffs)
        ### Initial values of equalization bias obtained as the last values after preprocessing one utterance four times
        self.equalization_bias[ 0] =  2.04312747
        self.equalization_bias[ 1] =  0.89522858
        self.equalization_bias[ 2] = -0.04273186
        self.equalization_bias[ 3] =  0.16099903
        self.equalization_bias[ 4] = -1.75399029
        self.equalization_bias[ 5] = -1.59060148
        self.equalization_bias[ 6] = -0.88533012
        self.equalization_bias[ 7] = -1.16879881
        self.equalization_bias[ 8] = -0.40637358
        self.equalization_bias[ 9] = -0.36991901
        self.equalization_bias[10] =  0.14560355
        self.equalization_bias[11] = -0.35795923
        self.equalization_bias[12] = -0.46702072


        A=10.0; B=1.0; C=5.0; D=1.5

        for k in range(self.num_coeffs):
            ### DCT-I
            factor = k * constants.pi / (self.num_filters - 1)
            for i in range(self.num_filters):
                self.table[k, i] = np.cos(factor * i)
            ### DCT-II
            #factor = k * constants.pi / self.num_filters
            #for i in range(self.num_filters):
            #    self.table[k] += np.cos(factor * (i + 0.5))
            # At the end self.table[0] should contain an array of all ones.
            #
            # We assume self.num_coeffs is 13, so
            #self.liftering_weights[k] = 1.0 + 0.5 * self.liftering_window_length * np.sin(k * constants.pi / self.liftering_window_length)
            self.liftering_weights[k] = A / (B + C * np.exp(-k / D))

    def get_num_coeffs(self):
        return self.num_coeffs

    def compute(self, fb):
        return np.dot(self.table, fb)

    def liftering(self, cc):
        for i in range(self.num_coeffs):
            cc[i] = cc[i] * self.liftering_weights[i]

    def equalization(self, cc, logE):
        #weighting_parameter = min(1.0, max(0.0, logE - 211.0 / 64.0)) # From ETSI preprocessor, but it uses a different value of logE
        #weighting_parameter = min(1.0, max(0.0, logE - 4.0))
        weighting_parameter = min(1.0, max(0.0, logE - 13.0)) # 13.0 implies that equalization_bias will be updated only when the utterance has enough energy.
        step_size = self.equalization_lambda * weighting_parameter
        for i in range(self.num_coeffs):
            diff = cc[i] - self.equalization_bias[i]
            self.equalization_bias[i] += step_size * diff
            cc[i] = diff


    def view_dct(self):

        import matplotlib.pyplot as plt

        for i in range(self.table.shape[0]):
            plt.plot(self.table[i])
        plt.grid()
        plt.show()
