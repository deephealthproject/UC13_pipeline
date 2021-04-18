import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

def model_1(input_shape, output_shape, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 1], strides = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 64, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 1])
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.ReLu(eddl.Conv2D(layer, 96, kernel_size = [3, 3], padding = 'same'))
        layer = eddl.Reshape(layer, [-1])
        layer = eddl.Activation(eddl.Dense(layer, 64), 'relu')
        layer = eddl.Softmax(eddl.Dense(layer, 2))

        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    else:
        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-6, momentum = 0.9), # previously lr was 1.0e-5
            lo = ["categorical_cross_entropy"],
            me = ["categorical_accuracy"],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            init_weights = initialize
            )
    eddl.summary(net)

    return net