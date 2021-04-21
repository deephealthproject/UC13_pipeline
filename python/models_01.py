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
        layer = eddl.GaussianNoise(layer, 0.25)
        do_pool = False
        for num_filters in [30, 60, 120, 240]:
            if do_pool:
                layer = eddl.MaxPool2D(layer, pool_size = [2, 2], strides = [2, 2])
            layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            #layer = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer, num_filters, kernel_size = [3, 3], padding = 'same'), affine = True))
            do_pool = True
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1000), affine = True))
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  500), affine = True))
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


if __name__ == '__main__':
    
    net = model_classifier_1a(input_shape = [1, 23, 20], num_classes = 2)
