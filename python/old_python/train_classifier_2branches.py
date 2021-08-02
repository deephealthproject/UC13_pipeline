import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils_eeg import DataGenerator
from models_02 import model_classifier_1a, model_classifier_1b, model_classifier_2a


if __name__ == '__main__':

    starting_epoch = 0
    epochs = 10
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
        elif param == '--starting-epoch':
            starting_epoch = int(sys.argv[i+1])
        elif param == '--epochs':
            epochs = int(sys.argv[i+1])
        elif param == '--batch-size':
            batch_size = int(sys.argv[i+1])
        elif sys.argv[i] == '--index':
            index_filenames.append(sys.argv[i+1])

    if len(index_filenames) == 0:
        raise Exception('Nothing can be done without data, my friend!')


    dg = DataGenerator(index_filenames, 
                       batch_size = batch_size,
                       do_shuffle = True,
                       in_training_mode = True,
                       balance_classes = True,
                       verbose = 1)

    x, y, t = dg[0]
    input_shape = (1,) + x.shape[1:]
    #print(input_shape)
    if model_id == '1a':
        net = model_classifier_1a(input_shape, num_classes = 2, filename = model_filename)
    elif model_id == '1b':
        net = model_classifier_1b(input_shape, num_classes = 4, filename = model_filename)
    elif model_id == '2a':
        net = model_classifier_2a(input_shape, num_classes = 2, filename = model_filename)

    else:
        raise Exception('You have to indicate a model id!')


    log_file = open(f'log/model_classifier_{model_id}.log', 'a')

    for epoch in range(starting_epoch, epochs):
        print()
        print(f'epoch: {epoch} of {epochs-1}', 'num batches:', len(dg))
        eddl.reset_loss(net)
        c0 = 0
        c1 = 0
        for j in range(len(dg)):
            x, y, t = dg[j]
            x = numpy.expand_dims(x, axis = 1)
            indices = list(range(len(x)))
            x0 = x[:,:,:,:8]
            x1 = x[:,:,:,8:]
            x0 = Tensor.fromarray(x0)
            x1 = Tensor.fromarray(x1)
            #_y_ = numpy.zeros([len(y), 2])
            #_y_[y == 0, 0] = 1
            #_y_[y == 1, 1] = 1
            #y = Tensor.fromarray(_y_)

            _y_ = numpy.zeros([len(y), 4])#, dtype=int)
            _y_[y == 0, 0] = 1
            _y_[y == 1, 1] = 1
            _y_[y == 2, 2] = 1
            _y_[y == 3, 3] = 1
            _y_ = _y_.reshape((len(y), 1, 4))
            y = Tensor.fromarray(_y_)
            eddl.train_batch(net, [x0, x1], [y], indices = indices)
            eddl.print_loss(net, j)
            print('\r', end='')

        log_file.write("epoch %d   softmax_cross_entropy %g   categorical_accuracy %g\n" % (epoch, eddl.get_losses(net)[0], eddl.get_metrics(net)[0]))
        log_file.flush()
        #eddl.save_net_to_onnx_file(net, f'models/model_classifier_{model_id}-{epoch}.onnx')
        eddl.save(net, f'models/model_classifier_{model_id}-{epoch}.eddl')
        dg.on_epoch_end()
    log_file.close()
