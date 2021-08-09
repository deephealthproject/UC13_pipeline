import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor


def model_classifier_1a(input_shape, num_classes, filename = None):

    if filename is None or filename.endswith('.eddl'):

        freq_in_ = eddl.Input(input_shape[:-1] + (8,))
        td_in_ = eddl.Input(input_shape[:-1] + (6,))

        layer1 = freq_in_
        layer1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer1, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer1 = eddl.Dropout(layer1, 0.2)
        layer1 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer1, 64, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer1 = eddl.Dropout(layer1, 0.2)

        layer2 = td_in_
        layer2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer2, 32, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer2 = eddl.Dropout(layer2, 0.2)
        layer2 = eddl.ReLu(eddl.BatchNormalization(eddl.Conv2D(layer2, 64, kernel_size = [3, 3], padding = 'same'), affine = True))
        layer2 = eddl.Dropout(layer2, 0.2)

        layer = eddl.Concat([layer1, layer2], 2)

        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer, 1000), affine = True))
        layer = eddl.Dropout(layer, 0.2)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  500), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([freq_in_, td_in_], [out_])
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


if __name__ == '__main__':
    
    net = model_classifier_1b(input_shape = (1, 252, 14), num_classes = 2)
