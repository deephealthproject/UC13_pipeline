import os
import sys
import numpy
import pickle

from sklearn import metrics

from machine_learning import KMeans, kmeans_load

from data_utils_eeg import SimpleDataGenerator
from generate_labels import index_to_label, label_to_index



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


    dg = SimpleDataGenerator(index_filenames, verbose = 0)

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

        num_labels = len(label_to_index)
        dim = kmeans.cluster_centers_.shape[1]

        # initialisation
        counters = list()
        for l in range(num_labels):
            counters.append([1] * kmeans.n_clusters)

        # accumulation
        while True:
            x, l = dg.next()
            if x is None: break
            y_pred = kmeans.predict(x)
            for cluster_id in y_pred:
                counters[l][cluster_id] += 1

        # normalisation to obtain p(l, c)
        s = 0
        for l in range(num_labels):
            s += sum(counters[l])
        for l in range(num_labels):
            for c in range(kmeans.n_clusters):
                counters[l][c] /= s

        # saving
        with open(f'models/conditionals_{num_labels}_{kmeans.n_clusters}_{dim}.pickle', 'wb') as f:
            pickle.dump(counters, f)
            f.close()

        for l in range(num_labels):
            print("%-10s" % index_to_label[l], end = ' ')
            for c in range(len(counters[l])):
                print(' %.6f' % counters[l][c], end = ' ')
            print()

    elif task == 'evaluate':

        kmeans = kmeans_load('models/kmeans.14')
        num_labels = len(label_to_index)
        dim = kmeans.cluster_centers_.shape[1]

        with open(f'models/conditionals_{num_labels}_{kmeans.n_clusters}_{dim}.pickle', 'rb') as f:
            joint_probabilities = pickle.load(f)
            f.close()
        joint_probabilities = numpy.array(joint_probabilities)
        conditionals = joint_probabilities.copy()

        #### to correct the imbalance, not theoretically correct
        # Due to the strong imbalance of target labels in the dataset, first do a normalisation
        # per label in order to compensate the imbalance
        for l in range(conditionals.shape[0]):
            conditionals[l, :] = joint_probabilities[l, :] / joint_probabilities[l, :].sum()
        #### to correct the imbalance, not theoretically correct

        #### normalisation to work with p(label | cluster)
        for c in range(conditionals.shape[1]):
            conditionals[:, c] = conditionals[:, c] / conditionals[:, c].sum()
        #### normalisation to work with p(label | cluster)

        l_true = list()
        l_pred = list()
        while True:
            x, l = dg.next()
            if x is None: break
            c_pred = kmeans.predict(x)
            probs = numpy.zeros(num_labels)
            for c in c_pred:
                probs += conditionals[:, c] # .T
            #
            l_true.append(l)
            l_pred.append(probs.argmax())

        print(metrics.confusion_matrix(l_true, l_pred, labels = range(len(index_to_label))))
        print()
        print(metrics.classification_report(l_true, l_pred, target_names = index_to_label))


    elif task == 'check':

        i = 0
        for x in kmeans.cluster_centers_:
            if sum(numpy.isnan(x)) > 0:
                print('NaN')
            print("%4d" % i, " ".join("{:10.6f}".format(_) for _ in x))
            i += 1

        counter = 0
        nan_counters = [0] * 14 # kmeans.cluster_centers_.shape[1]
        while True:
            x, l = dg.next()
            if x is None: break
            for i in range(len(nan_counters)):
                nan_counters[i] +=  sum(numpy.isnan(x[:, i]))
            counter += 1
            print("\r %s  %20d" % (" ".join("{:8d}".format(_) for _ in nan_counters), counter), end = ' ')
        print()
        print(f'NaN counter {sum(nan_counters)}')
