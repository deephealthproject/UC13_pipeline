import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils_eeg import DataGenerator, SequenceDataGenerator
from models_01 import model_classifier_1a, model_classifier_2a, model_classifier_3a, model_classifier_3b


if __name__ == '__main__':

    starting_epoch = 0
    epochs = 10
    batch_size = 100
    model_filename = None
    model_id = '1a'
    index_filenames = list()
    index_validation = list()
    gpu = [1]
    for i in range(1, len(sys.argv)):
        param = sys.argv[i]
        if param == '--model-filename':
            model_filename = sys.argv[i+1]
        elif param == '--model':
            model_id = sys.argv[i+1]
        elif param == '--starting-epoch':
            starting_epoch = int(sys.argv[i+1])
        elif param == '--epochs':
            epochs = int(sys.argv[i+1])
        elif param == '--batch-size':
            batch_size = int(sys.argv[i+1])
        elif sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])
        elif sys.argv[i] == '--index-val':
            index_validation.append(sys.argv[i+1])
        elif sys.argv[i] == '--gpu':
            i = int(sys.argv[i+1])
            if i == 0:
                gpu = [1]
            elif i == 1:
                gpu = [0, 1]

    if len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')


    dg = SequenceDataGenerator(index_filenames, 
                       batch_size = batch_size,
                       sequence_length = 50,
                       do_shuffle = True,
                       in_training_mode = True,
                       do_standard_scaling = True,
                       verbose = 1)

    dg_val = None
    if len(index_validation) != 0:
        dg_val = SequenceDataGenerator(index_validation,
                                        batch_size = batch_size,
                                        sequence_length = 50,
                                        do_shuffle = True,
                                        in_training_mode = True,
                                        do_standard_scaling = True,
                                        verbose = 1)

    x, y, t = dg[0]
    input_shape = x.shape[2:]

    if model_id == '3a':
        net = model_classifier_3a(input_shape, num_classes = 4, filename = model_filename)
    elif model_id == '3b':
        net = model_classifier_3b(input_shape, num_classes = 4, filename = model_filename)
    else:
        raise Exception('You have to indicate a model id!')


    log_file = open(f'log/model_classifier_{model_id}.log', 'a')

    for epoch in range(starting_epoch, epochs):
        print()
        print(f'epoch: {epoch} of {epochs}', 'num batches:', len(dg))
        eddl.reset_loss(net)

        for j in range(len(dg)):
            x, y, t = dg[j]
            
            indices = list(range(len(x)))
            x = Tensor.fromarray(x)
            _y_ = numpy.zeros([len(y), 4])#, dtype=int)
            _y_[y == 0, 0] = 1
            _y_[y == 1, 1] = 1
            _y_[y == 2, 2] = 1
            _y_[y == 3, 3] = 1
            _y_ = _y_.reshape((len(y), 1, 4))
            y = Tensor.fromarray(_y_)

            eddl.train_batch(net, [x], [y], indices = indices)
            eddl.print_loss(net, j)
            print('\r', end='')
            #print(eddl.get_losses(net))
            #print(eddl.get_metrics(net))

        print()

        log_file.write("epoch %d   cross_entropy %g   categorical_accuracy %g\n" % (epoch, eddl.get_losses(net)[0], -1)) # eddl.get_metrics(net)[0]))
        #log_file.flush()
        #eddl.save_net_to_onnx_file(net, f'models/model_classifier_{model_id}-{epoch}.onnx')
        eddl.save(net, f'models/model_classifier_{model_id}-{epoch}.eddl')
        dg.on_epoch_end()

        if (dg_val is not None):
            # Validation
            Y_true = list()
            Y_pred = list()
            for j in range(len(dg_val)):
                x, y_true, t = dg_val[j]
                x = Tensor.fromarray(x)
                (y_pred, ) = eddl.predict(net, [x])
                y_pred = y_pred.getdata()
                Y_true.append(y_true)
                Y_pred.append(y_pred.argmax(axis = 1))
            y_true = numpy.hstack(Y_true) * 1.0
            y_pred = numpy.hstack(Y_pred) * 1.0
            print('validation accuracy epoch %d = %g' % ( epoch, sum(y_true == y_pred) / len(y_true)))
            log_file.write('validation accuracy epoch %d = %g\n' % ( epoch, sum(y_true == y_pred) / len(y_true))) # eddl.get_metrics(net)[0]))

        log_file.flush()

    log_file.close()
