import os
import sys
import numpy

from sklearn.metrics import classification_report, confusion_matrix


from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils_eeg import SequenceDataGenerator
from models_01 import model_classifier_3a, model_classifier_3b


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


    dg = SequenceDataGenerator(index_filenames, 
                       batch_size = batch_size,
                       sequence_length = 50,
                       do_shuffle = True,
                       in_training_mode = False,
                       do_standard_scaling = True,
                       #balance_classes = True,
                       verbose = 1)

    x, y, t = dg[0]
    input_shape = x.shape[2:]
    if model_id == '3a':
        net = model_classifier_3a(input_shape, num_classes = 4, filename = model_filename)
    elif model_id == '3b':
        net = model_classifier_3b(input_shape, num_classes = 4, filename = model_filename)
    else:
        raise Exception('You have to indicate a model id!')

    # Train a batch
    x = Tensor.fromarray(x)
    _y_ = numpy.zeros([len(y), 4])#, dtype=int)
    _y_[y == 0, 0] = 1
    _y_[y == 1, 1] = 1
    _y_[y == 2, 2] = 1
    _y_[y == 3, 3] = 1
    _y_ = _y_.reshape((len(y), 1, 4))
    y = Tensor.fromarray(_y_)
    eddl.train_batch(net, [x], [y])


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
    #print('recall    = ', sum(numpy.logical_and(y_true, y_pred)) / (1.0e-6 + sum(y_true)))
    #print('precision = ', sum(numpy.logical_and(y_true, y_pred)) / (1.0e-6 + sum(y_pred)))
    #print(y_true.shape, y_pred.shape)
    print(confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3]))
    print(classification_report(y_true, y_pred, target_names = ['inter-ictal', 'ictal', 'pre-ictal', 'post-ictal'])) 
