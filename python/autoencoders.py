import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

def model_1a(input_shape, output_shape, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
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

def model_1b(input_shape, output_shape, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [7, 5], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [7, 5], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [5, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [5, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 128, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 128, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [5, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [5, 3], padding = 'same'))
        layer = eddl.UpSampling(layer, size = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [7, 5], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [7, 5], padding = 'same'))
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
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 1], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        
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
