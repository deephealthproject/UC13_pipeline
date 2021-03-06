import os
import sys
import numpy
import pickle
import math

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils import DataGenerator
from autoencoders import model_1a, model_1b, model_2


if __name__ == '__main__':

    starting_epoch = 0
    epochs = 10
    batch_size = 70
    model_filename = None
    model_id = '2'
    dirnames = list()
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
        elif param == '--dirname':
            dirnames.append(sys.argv[i+1])

    if len(dirnames) == 0:
        dirnames = ['../UC13/clean_signals/chb01/test']

    dg = DataGenerator(dirnames,
                        batch_size = batch_size,
                        do_shuffle = False,
                        do_preemphasis = True,
                        n_processes = 16)

    x, y = dg[0]
    input_shape = (1,) +  x.shape[1:]
    if model_id == '1a':
        net = model_1a(input_shape, input_shape, filename = model_filename)
    elif model_id == '1b':
        net = model_1a(input_shape, input_shape, filename = model_filename)
    elif model_id == '2':
        net = model_2(input_shape, input_shape, filename = model_filename)
    else:
        raise Exception('You have to indicated a model id!')

    
    distances_class_zero = []
    distances_class_one = []

    for x, y in dg:
        x = numpy.expand_dims(x, axis = 1)
        indices = list(range(len(x)))
        x = Tensor.fromarray(x)
        
        # Compute Euclidean distance between original signal and prediction
        prediction = eddl.predict(net,[x])
        for i in indices:
            original = x.select([str(i),":",":",":"])
            pred = prediction[0].select([str(i),":",":",":"])
            dist = original.sub(pred).sqr()
            dist = math.sqrt(numpy.sum(dist.getdata()))
            if y[i]==0:
                distances_class_zero.append(dist)
            if y[i]==1:
                distances_class_one.append(dist)

    with open('temp/dist_normal.pkl', 'wb') as f:
        pickle.dump(distances_class_zero, f)
    with open('temp/dist_seizure.pkl', 'wb') as f:
        pickle.dump(distances_class_one, f)
