import sys
import numpy
import math
from tqdm import tqdm
from data_utils_eeg import load_file
import matplotlib.pyplot as plt

def draw_histogram(x, title="Histogram"):
    """ Draw an histogram"""

    fig = plt.figure(figsize=(10,7))
    n, bins, patches = plt.hist(x, bins=50)
    plt.grid()
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.title(title + ' Histogram')
    #plt.show()
    plt.savefig('etc/' + title + '.png')


if __name__ == '__main__':
    #
    index_filenames = list()
    #
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])
    #
    filenames = list()
    for index_fname in index_filenames:
        with open(index_fname, 'r') as f:
            for l in f:
                filenames.append(l.strip())
            f.close()
    #
    whole_data = list()
    for fname in filenames:
        if fname[0] != '#':
                x_fbank, x_stats, labels, timestamp = load_file(fname, verbose = 1)
                x = numpy.concatenate((x_fbank, x_stats), axis=2)
                whole_data.append(x)
    #
    # Create a list to store each feature data
    x_fbank = list()
    x_mean = list()
    x_std = list()
    x_kurt = list()
    x_skew = list()
    x_mob = list()
    x_comp = list()
    for x in tqdm(whole_data):
        x_fbank += x[:,:,:8].flatten().tolist()
        x_mean += x[:,:,8].tolist()
        x_std += x[:,:,9].tolist()
        x_kurt += x[:,:,10].tolist()
        x_skew += x[:,:,11].tolist()
        x_mob += x[:,:,12].tolist()
        x_comp += x[:,:,13].tolist()
    #
    draw_histogram(x_fbank, title="Fbank")
    draw_histogram(x_mean, title="Mean")
    draw_histogram(x_std, title="Standard Deviation")
    draw_histogram(x_kurt, title="Kurtosis")
    draw_histogram(x_skew, title="Skewness")
    draw_histogram(x_mob, title="Mobility")
    draw_histogram(x_comp, title="Complexity")