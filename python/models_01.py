import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

def model_classifier_1a(input_shape, num_classes, filename = None):

    if filename is None:
        in_ = eddl.Input(input_shape)
        layer = in_
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 16, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 64, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, 64, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1024), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1024), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))

        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    else:
        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    eddl.build(
            net,
            o = eddl.sgd(lr = 1.e-4, momentum = 0.9),
            lo = ['softmax_cross_entropy'],
            me = ['accuracy'], # ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = [1], mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )
    eddl.summary(net)

    return net
