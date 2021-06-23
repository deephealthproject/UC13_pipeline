import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

def model_classifier_1a(input_shape, num_classes, filename = None):

    if filename is None or filename.endswith('.eddl'):

        in_ = eddl.Input(input_shape)

        layer = in_
        #
        #layer = eddl.GaussianNoise(layer, 0.25)
        do_pool = False
        for num_filters in [16, 32]:
            if do_pool:
                layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
            layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            #layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            do_pool = True

        for num_filters in [64, 96]:
            if do_pool:
                layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
            layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            #layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            do_pool = True
        
        
        #layer = eddl.GlobalMaxPool2D(layer)
        layer = eddl.Flatten(layer)
        #layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1000), affine = True))
        #layer = eddl.Dropout(layer, 0.4)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  500), affine = True))
        #layer = eddl.Dropout(layer, 0.3)
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True

    elif filename.endswith('.onnx'):

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:
        raise Exception('unexpected file format.')

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-3, momentum = 0.9),
            lo = ['softmax_cross_entropy'],
            me = ['accuracy'], # ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )

    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)

    return net


def model_classifier_2a(input_shape, num_classes, filename = None):

    if filename is None or filename.endswith('.eddl'):

        in_ = eddl.Input(input_shape)

        layer = in_
        #
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 2000), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 2000), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1000), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True

    elif filename.endswith('.onnx'):

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:
        raise Exception('unexpected file format.')

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-5, momentum = 0.9),
            lo = ['softmax_cross_entropy'],
            me = ['accuracy'], # ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )

    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)

    return net


def model_classifier_3a(input_shape, num_classes, filename = None):

    if filename is not None and filename.endswith('.onnx'):

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:

        in_ = eddl.Input(input_shape)

        layer = in_
        #
        #layer = eddl.Dense(layer, 32)
        layer = eddl.ReLu(eddl.LSTM(layer, 512))
        #layer = eddl.ReLu(eddl.GRU(layer, 64))
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  512), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine = True))
        #layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        layer = eddl.Sigmoid(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-4, momentum = 0.9),
            #lo = ['softmax_cross_entropy'],
            lo = ['cross_entropy'],
            me = ['categorical_accuracy'], # ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = [0, 1], mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )
    
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net


def model_classifier_3b(input_shape, num_classes, filename = None):

    if filename is not None and filename.endswith('.onnx'):

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:

        in_ = eddl.Input(input_shape)

        layer = in_
        #
        #layer = eddl.Dense(layer, 32)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 64, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 128, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.GlobalMaxPool2D(layer)
        layer = eddl.Flatten(layer)

        #layer = eddl.ReLu(eddl.LSTM(layer, 64))
        #layer = eddl.ReLu(eddl.GRU(layer, 64))
        #layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-3, momentum = 0.9),
            lo = ['softmax_cross_entropy'],
            me = ['categorical_accuracy'], # ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = [0, 1], mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )
    
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)

    return net

if __name__ == '__main__':
    
    #net = model_classifier_1a(input_shape = [1, 23, 20], num_classes = 2)
    net = model_classifier_3b(input_shape = [100, 50, 21, 14], num_classes = 4)
