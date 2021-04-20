import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils_fbank import DataGenerator
from models_01 import model_classifier_1a


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


    dg = DataGenerator(index_filenames, batch_size = batch_size, n_processes = 16)

    x, y, t = dg[0]
    input_shape = x.shape[1:]
    if model_id == '1a':
        net = model_classifier_1a(input_shape, num_classes = 2, filename = model_filename)
    #elif model_id == '1b':
    #    net = model_1a(input_shape, input_shape, filename = model_filename)
    #elif model_id == '2':
    #    net = model_2(input_shape, input_shape, filename = model_filename)
    else:
        raise Exception('You have to indicated a model id!')


    log_file = open(f'log/model_classifier_{model_id}.log', 'a')

    for epoch in range(starting_epoch, epochs):
        print()
        print('epoch:', epoch, 'num batches:', len(dg))
        eddl.reset_loss(net)
        c0 = 0
        c1 = 0
        for j in range(len(dg)):
            x, y, t = dg[j]
            #x = numpy.expand_dims(x, axis = 1)
            indices = list(range(len(x)))
            x = Tensor.fromarray(x)
            c0 += len(y)
            c1 += sum(y == 1)
            _y_ = numpy.zeros([len(y), 2])
            _y_[y == 0, 0] = 1
            _y_[y == 1, 1] = 1
            y = Tensor.fromarray(_y_)
            eddl.train_batch(net, [x], [y], indices = indices)
            eddl.print_loss(net, j)
            print('%g \r' % (c1 / c0), end = '')
            #j += 1

        log_file.write("epoch %d   softmax_cross_entropy %g   categorical_accuracy %g\n" % (epoch+1, eddl.get_losses(net)[0], eddl.get_metrics(net)[0]))
        log_file.flush()
        eddl.save_net_to_onnx_file(net, f'models/model_classifier_{model_id}-{epoch}.onnx')
        dg.on_epoch_end()
    log_file.close()
