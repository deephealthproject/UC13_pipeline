import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils import DataGenerator

def model_1(input_shape, output_shape, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1]) #, padding = 'same')
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1]) #, padding = 'same')
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.Conv2D(layer, 1, kernel_size = [3, 3], padding = 'same')

        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    else:
        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-6, momentum = 0.9), # previously lr was 1.0e-5
            lo = ["mse"],
            me = ["mean_absolute_error"],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            init_weights = initialize
            )
    eddl.summary(net)

    return net

def model_2(input_shape, output_shape, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1]) #, padding = 'same')
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1]) #, padding = 'same')
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.Conv2D(layer, 1, kernel_size = [3, 1], padding = 'same')

        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    else:
        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-6, momentum = 0.9),
            lo = ["mse"],
            me = ["mean_absolute_error"],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            init_weights = initialize
            )
    eddl.summary(net)

    return net



if __name__ == '__main__':

    starting_epoch = 0
    epochs = 10
    model_filename = None
    model_id = 2
    for i in range(1, len(sys.argv)):
        param = sys.argv[i]
        if param == '--model-filename':
            model_filename = sys.argv[i+1]
        elif param == '--model':
            model_id = int(sys.argv[i+1])
        elif param == '--starting-epoch':
            starting_epoch = int(sys.argv[i+1])
        elif param == '--epochs':
            epochs = int(sys.argv[i+1])

    dg = DataGenerator(['clean_signals/chb01'], batch_size = 70, do_shuffle = True, n_processes = 16)
    #dg = DataGenerator(['clean_signals/testing'], batch_size = 10, do_shuffle = True, n_processes = 16)

    '''
    total = 0
    counter = 0
    for x, y in dg:
        total += x.sum(axis = 0)
        counter += len(x)
    total /= counter
    print(total)
    '''

    x, y = dg[0]
    '''
    print('x', x.shape, x.ptp(), x.min(), x.max())
    print('y', y.shape)
    '''

    input_shape = (1,) +  x.shape[1:]
    if model_id == 1:
        net = model_1(input_shape, input_shape, filename = model_filename)
    elif model_id == 2:
        net = model_2(input_shape, input_shape, filename = model_filename)
    else:
        raise Exception('You have to indicated a model id!')


    log_file = open(f'log/model_{model_id}.log', 'a')

    for epoch in range(starting_epoch, epochs):
        print()
        print('epoch:', epoch, 'num batches:', len(dg))
        eddl.reset_loss(net)
        j = 0
        for x, y in dg:
        #for i in range(10):
        #    x , y = dg[i]
            x = numpy.expand_dims(x, axis = 1)
            indices = list(range(len(x)))
            x = Tensor.fromarray(x)
            eddl.train_batch(net, [x], [x], indices = indices)
            eddl.print_loss(net, j)
            print('\r', end = '')
            j += 1

        log_file.write("epoch %d   mse %g  mae %g\n" % (epoch+1, eddl.get_losses(net)[0], eddl.get_metrics(net)[0]))
        log_file.flush()
        eddl.save_net_to_onnx_file(net, f'models/model_{model_id}-{epoch}.onnx')
        dg.on_epoch_end()
    log_file.close()

    #eddl.evaluate(net, [x_test], [y_test])
