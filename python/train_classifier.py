import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils import DataGenerator
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

    dg = DataGenerator(['../UC13/clean_signals/chb03/train'], batch_size = 100, do_shuffle = True, 
                        n_processes = 16, exclude_seizures = True,
                        in_training_mode = True)

    x, y = dg[0]
    input_shape = (1,) +  x.shape[1:]
    if model_id == '1':
        net = model_1(input_shape, input_shape, filename = model_filename)
    else:
        raise Exception('You have to indicate a model id!')

    log_file = open(f'log/model_{model_id}_classifier.log', 'a')

    for epoch in range(starting_epoch, epochs):
        print()
        print('epoch:', epoch, 'num batches:', len(dg))
        eddl.reset_loss(net)
        j = 0
        
        for x, y in dg:
            x = numpy.expand_dims(x, axis = 1)
            indices = list(range(len(x)))
            x = Tensor.fromarray(x)
            eddl.train_batch(net, [x], [x], indices = indices)
            eddl.print_loss(net, j)
            print('\r', end = '')
            j += 1

        log_file.write("epoch %d   bce %g  acc %g\n" % (epoch+1, eddl.get_losses(net)[0], eddl.get_metrics(net)[0]))
        log_file.flush()
        eddl.save_net_to_onnx_file(net, f'models/classifiers/model_{model_id}-{epoch}.onnx')
        dg.on_epoch_end()
    log_file.close()