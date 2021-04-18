import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils import DataGenerator
from data_utils import decompress_pickle
from data_utils import load_data
from classifiers import model_1


if __name__ == '__main__':

    starting_epoch = 0
    epochs = 25
    model_filename = None
    model_id = '1'
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


    batch_size = 128
    x, y = load_data(['../../UC13/clean_signals/chb01/train'], exclude_seizures = False)

    input_shape = (1,) +  x.shape[1:]
    if model_id == '1':
        net = model_1(input_shape, input_shape, filename = model_filename)
    else:
        raise Exception('You have to indicate a model id!')

    log_file = open(f'log/model_{model_id}_classifier.log', 'a')

    x_train = numpy.expand_dims(x, axis = 1)
    x_train = Tensor.fromarray(x_train)
    y_train = Tensor.fromarray(y)

    eddl.fit(net, [x_train], [y_train], batch_size, epochs)

    print("Evaluating the model...")
    x_test, y_test = load_data(['../../UC13/clean_signals/chb01/test'], exclude_seizures = False)
    x_test = numpy.expand_dims(x_test, axis = 1)
    x_test = Tensor.fromarray(x_test)
    y_test = Tensor.fromarray(y_test)
    
    eddl.evaluate(net, [x_test], [y_test])