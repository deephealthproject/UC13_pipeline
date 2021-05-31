import os
import sys
import numpy


from data_utils_eeg import SimpleDataGenerator

from machine_learning import KMeans, kmeans_load


if __name__ == '__main__':

    models_filename = 'models/kmeans.14'
    task = 'evaluate'
    index_filenames = list()
    for i in range(1, len(sys.argv)):
        param = sys.argv[i]
        if sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])
        elif sys.argv[i] == '--task':
            task = sys.argv[i+1]

    if task not in ['training', 'evaluate', 'compute-empirical-distribution']:
        raise Exception(f'Unrecognized task {task}')


    if len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')


    dg = SimpleDataGenerator(index_filenames, verbose = 1)

    if task == 'training':

        if os.path.exists(models_filename):
            print()
            print(f'The file {models_filename} already exists, please, remove it before invoking this program again to do the computation!')
            print()
            sys.exit(0)

        kmeans = KMeans(n_clusters = 50, modality = 'original-k-Means')

        # generate a list of samples that contains 'kmeans.n_clusters' samples a least
        X = list()
        while len(X) < kmeans.n_clusters:
            x, l = dg.next()
            if x is None or len(x) == 0: break
            X += [_ for _ in x]

        # do initialization
        kmeans.original_k_means_init(X)

        # sequentially process all the remaining samples
        chars = '|/-\\'
        i = 0
        while True:
            #X, L = dg.next_block()
            x, l = dg.next()
            if x is None or len(x) == 0: break
            kmeans.original_k_means_iteration(x)
            print(chars[i],  end = '\r', flush = True)
            i = (i + 1) % 4
        
        kmeans.save('models/kmeans.14')

    elif task == 'compute-empirical-distribution':
        kmeans = kmeans_load('models/kmeans.14')

    elif task == 'evaluate':
        '''
        kmeans = kmeans_load('models/kmeans.14')
        i = 0
        for x in kmeans.cluster_centers_:
            if sum(numpy.isnan(x)) > 0:
                print('NaN')
            #print("%4d" % i, " ".join("{:10.6f}".format(_) for _ in x))
            i += 1
        '''
        counter = 0
        nan_counters = [0] * 14 # kmeans.cluster_centers_.shape[1]
        while True:
            x, l = dg.next()
            if x is None: break
            #if sum(numpy.isnan(x.any())) > 0:
            for i in range(len(nan_counters)):
                nan_counters[i] +=  sum(numpy.isnan(x[:, i]))
            counter += 1
            print("\r %s  %20d" % (" ".join("{:8d}".format(_) for _ in nan_counters), counter), end = ' ')
        print()
        print(f'NaN counter {sum(nan_counters)}')
