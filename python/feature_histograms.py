import sys
import numpy
import math
from tqdm import tqdm
from data_utils_eeg import load_file
import matplotlib.pyplot as plt

def draw_histogram(x, bins, title="Histogram"):
    """ Draws an histogram"""

    fig = plt.figure(figsize=(10,7))
    n, bins, patches = plt.hist(x, bins=bins)
    plt.grid()
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.title(title + ' Histogram')
    #plt.show()
    plt.savefig('etc/' + title + '.png')
    print(title, "Histogram DONE")


class FeatureHistograms:
    """
    Class for checking the scaling and plotting histograms
    """
    def __init__(self, index_filenames = None):

        self.filenames = list()
        for fname in index_filenames:
            with open(fname, 'r') as f:
                for l in f:
                    self.filenames.append(l.strip())
                f.close()
        #
        self.whole_data = list()
        for fname in self.filenames:
            if fname[0] != '#':
                x_fbank, x_stats, labels, timestamp = load_file(fname, verbose = 1)
                x = numpy.concatenate((x_fbank, x_stats), axis=2)
                self.whole_data.append(x)
        #

        # Create a list to store each feature data
        x_fbank = list()
        x_mean = list()
        x_std = list()
        x_kurt = list()
        x_skew = list()
        x_mob = list()
        x_comp = list()
        for x in tqdm(self.whole_data):
            x_fbank += x[:,:,:8].flatten().tolist()
            x_mean += x[:,:,8].tolist()
            x_std += x[:,:,9].tolist()
            x_kurt += x[:,:,10].tolist()
            x_skew += x[:,:,11].tolist()
            x_mob += x[:,:,12].tolist()
            x_comp += x[:,:,13].tolist()
        #

        print("Values under 0.5 in fbank:", counter)
        print("X_fbank min and max values: ", min(x_fbank), max(x_fbank))
        #draw_histogram(x_fbank, 50, title="Fbank")
        #draw_histogram(x_mean, "auto", title="Mean")
        #draw_histogram(x_std, "auto", title="Standard Deviation")
        #draw_histogram(x_kurt, "auto", title="Kurtosis")
        #draw_histogram(x_skew, "auto", title="Skewness")
        #draw_histogram(x_mob, "auto", title="Mobility")
        #draw_histogram(x_comp, "auto", title="Complexity")


        # Calculate means and stddevs
        fbank_means = []
        fbank_stddevs = []
        mean_means = []
        mean_stddevs = []
        std_means = []
        std_stddevs = []
        kurt_means = []
        kurt_stddevs = []
        skew_means = []
        skew_stddevs = []
        mob_means = []
        mob_stddevs = []
        comp_means = []
        comp_stddevs = []
        counts = []

        for x in tqdm(self.whole_data):
            x_fbank_scale = x[:,:,:8]
            x_mean_scale = x[:,:,8]
            x_std_scale = x[:,:,9]
            x_kurt_scale = x[:,:,10]
            x_skew_scale = x[:,:,11]
            x_mob_scale = x[:,:,12]
            x_comp_scale = x[:,:,13]
            
            fbank_means.append(x_fbank_scale.mean())
            fbank_stddevs.append(x_fbank_scale.std())
            mean_means.append(x_mean_scale.mean(axis=0))
            mean_stddevs.append(x_mean_scale.std(axis=0))
            std_means.append(x_std_scale.mean(axis=0))
            std_stddevs.append(x_std_scale.std(axis=0))
            kurt_means.append(x_kurt_scale.mean(axis=0))
            kurt_stddevs.append(x_kurt_scale.std(axis=0))
            skew_means.append(x_skew_scale.mean(axis=0))
            skew_stddevs.append(x_skew_scale.std(axis=0))
            mob_means.append(x_mob_scale.mean(axis=0))
            mob_stddevs.append(x_mob_scale.std(axis=0))
            comp_means.append(x_comp_scale.mean(axis=0))
            comp_stddevs.append(x_comp_scale.std(axis=0))
            counts.append(len(x))
        #
        self.fbank_mean = sum([m * c for m, c in zip(fbank_means, counts)])
        self.fbank_std = sum([s * c for s, c in zip(fbank_stddevs, counts)])
        self.fbank_mean /= sum(counts) # We use one mean and std for the 8 channels in frequency
        self.fbank_std /= sum(counts)
        print("Fbank Mean:", self.fbank_mean)
        print("Fbank Std:", self.fbank_std)

        
        self.td_stats_means = []    # Lists with time domain statistics means and stdevs
        self.td_stats_stddevs = []  # 0: mean, 1: std, 2: kurtosis, 3: skewness 4: Mobility 5: Complexity

        for ft_means, ft_stddevs in [(mean_means, mean_stddevs), (std_means, std_stddevs),
                                (kurt_means, kurt_stddevs), (skew_means, skew_stddevs),
                                (mob_means, mob_stddevs), (comp_means, comp_stddevs)]:

            mean = sum([m * c for m, c in zip(ft_means, counts)])
            std = sum([m * c for m, c in zip(ft_stddevs, counts)])
            mean /= sum(counts)
            std /= sum(counts)
            self.td_stats_means.append(mean)
            self.td_stats_stddevs.append(std)
        #
        print("TD Stats means:", self.td_stats_means)
        print("TD Stats stddevs:", self.td_stats_stddevs)

        # Scale data and plot histograms after scaling
        x_fbank_scaled = list()
        x_mean_scaled = list()
        x_std_scaled = list()
        x_kurt_scaled = list()
        x_skew_scaled = list()
        x_mob_scaled = list()
        x_comp_scaled = list()

        for x in tqdm(self.whole_data):
            for x_i in x:
                x_i = self.scale_data(x_i)
                x_fbank_scaled += x_i[:,:8].flatten().tolist()
                x_mean_scaled += x_i[:,8].tolist()
                x_std_scaled += x_i[:,9].tolist()
                x_kurt_scaled += x_i[:,10].tolist()
                x_skew_scaled += x_i[:,11].tolist()
                x_mob_scaled += x_i[:,12].tolist()
                x_comp_scaled += x_i[:,13].tolist()
        #
        print(len(x_fbank), len(x_fbank_scaled))
        print("Fbank scaled mean", numpy.array(x_fbank_scaled).mean())
        print("Fbank scaled std", numpy.array(x_fbank_scaled).std())

        td_means_scaled = []
        td_means_scaled.append(numpy.array(x_mean_scaled).mean())
        td_means_scaled.append(numpy.array(x_std_scaled).mean())
        td_means_scaled.append(numpy.array(x_kurt_scaled).mean())
        td_means_scaled.append(numpy.array(x_skew_scaled).mean())
        td_means_scaled.append(numpy.array(x_mob_scaled).mean())
        td_means_scaled.append(numpy.array(x_comp_scaled).mean())

        td_stddevs_scaled = []
        td_stddevs_scaled.append(numpy.array(x_mean_scaled).std())
        td_stddevs_scaled.append(numpy.array(x_std_scaled).std())
        td_stddevs_scaled.append(numpy.array(x_kurt_scaled).std())
        td_stddevs_scaled.append(numpy.array(x_skew_scaled).std())
        td_stddevs_scaled.append(numpy.array(x_mob_scaled).std())
        td_stddevs_scaled.append(numpy.array(x_comp_scaled).std())
        
        print("TD Stats means:", td_means_scaled)
        print("TD Stats stddevs:", td_stddevs_scaled)
        #draw_histogram(x_fbank_scaled, 50, title="Fbank Scaled")
        #draw_histogram(x_mean_scaled, "auto", title="Mean Scaled")
        #draw_histogram(x_std_scaled, "auto", title="Standard Deviation Scaled")
        #draw_histogram(x_kurt_scaled, "auto", title="Kurtosis Scaled")
        #draw_histogram(x_skew_scaled, "auto", title="Skewness Scaled")
        #draw_histogram(x_mob_scaled, "auto", title="Mobility Scaled")
        #draw_histogram(x_comp_scaled, "auto", title="Complexity Scaled")
        
    #

    def scale_data(self, X):
        '''
        Scales the data to mean zero and standard deviation 1.

        :param self  Reference to the current object.

        :param numpy.array  X   An array with the data.

        :return numpy.array X   An array with the scaled data.
        '''

        X[:,:8] = (X[:,:8] - self.fbank_mean) / self.fbank_std
        X[:,8] = (X[:,8] - self.td_stats_means[0]) / self.td_stats_stddevs[0]
        X[:,9] = (X[:,9] - self.td_stats_means[1]) / self.td_stats_stddevs[1]
        X[:,10] = (X[:,10] - self.td_stats_means[2]) / self.td_stats_stddevs[2]
        X[:,11] = (X[:,11] - self.td_stats_means[3]) / self.td_stats_stddevs[3]
        X[:,12] = (X[:,12] - self.td_stats_means[4]) / self.td_stats_stddevs[4]
        X[:,13] = (X[:,13] - self.td_stats_means[5]) / self.td_stats_stddevs[5]
        #
        return X
    


if __name__ == '__main__':
    #
    index_filenames = list()
    #
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])
    #
    ft_obj = FeatureHistograms(index_filenames)