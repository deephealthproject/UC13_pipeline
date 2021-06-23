import os
import sys
import numpy

from sklearn.metrics import classification_report, confusion_matrix


from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils_eeg import DataGenerator
from models_01 import model_classifier_1a, model_classifier_2a


if __name__ == '__main__':

    batch_size = 100
    model_filename = None
    model_id = '1a'
    index_filenames = list()
    for i in range(1, len(sys.argv)):
        param = sys.argv[i]
        if param == '--model-filename':
            model_filename = sys.argv[i+1]
        elif param == '--model':
            model_id = sys.argv[i+1]
        elif param == '--batch-size':
            batch_size = int(sys.argv[i+1])
        elif sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])

    if len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')


    dg = DataGenerator(index_filenames,
                        batch_size = batch_size,
                        balance_classes = False,
                        verbose = 1)

    x, y, t = dg[0]
    input_shape = (1,) + x.shape[1:]
    if model_id == '1a':
        net = model_classifier_1a(input_shape, num_classes = 4, filename = model_filename)
    #elif model_id == '1b':
    #    net = model_1a(input_shape, input_shape, filename = model_filename)
    elif model_id == '2a':
        net = model_classifier_2a(input_shape, num_classes = 4, filename = model_filename)
    else:
        raise Exception('You have to indicate a model id!')

    Y_true = list()
    Y_pred = list()
    for j in range(len(dg)):
        x, y_true, t = dg[j]
        x = Tensor.fromarray(x)
        (y_pred, ) = eddl.predict(net, [x])
        y_pred = y_pred.getdata()
        Y_true.append(y_true)
        Y_pred.append(y_pred.argmax(axis = 1))

    y_true = numpy.hstack(Y_true) * 1.0
    y_pred = numpy.hstack(Y_pred) * 1.0
    #print('sum(y_true) = ', sum(y_true))
    #print('sum(y_pred) = ', sum(y_pred))
    print('accuracy  = ', sum(y_true == y_pred) / len(y_true))
    print('recall    = ', sum(numpy.logical_and(y_true, y_pred)) / (1.0e-6 + sum(y_true)))
    print('precision = ', sum(numpy.logical_and(y_true, y_pred)) / (1.0e-6 + sum(y_pred)))
    print(y_true.shape, y_pred.shape)
    #print(confusion_matrix(y_true, y_pred, labels = [0, 1]))
    #print(classification_report(y_true, y_pred, target_names = ['normal', 'ictal']))
    print(confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3]))
    print(classification_report(y_true, y_pred, target_names = ['inter-ictal', 'ictal', 'pre-ictal', 'post-ictal'])) 

