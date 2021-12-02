from pyeddl import eddl


def create_model(model_id, 
                 input_shape,
                 num_classes, 
                 filename = None, 
                 gpus = [1]):

    """
        Builds and returns a model given a model id,
        the number of classes to use, a filename to load
        the model (optional) and the gpus to use.
    """

    if model_id == 'lstm':
        net = recurrent_LSTM(input_shape,
                             num_classes=num_classes,
                             filename=filename,
                             gpus = gpus)
                             
    elif model_id == 'gru':
        net = recurrent_GRU(input_shape,
                            num_classes=num_classes,
                            filename=filename,
                            gpus = gpus)

    else:
        raise Exception('You have to provide an existing model_id!')
    
    return net



def recurrent_LSTM(input_shape, num_classes, filename = None, gpus = [1]):

    if filename is not None and filename.endswith('.onnx'):
        # Load .onnx file if it is the case

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:
        # Create the model from scratch

        in_ = eddl.Input(input_shape)
        #
        layer = eddl.LSTM(in_, 256)
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    
    #print(f'{initialize=}')
    eddl.build(
            net,
            o = eddl.adam(lr = 1.0e-4),
            lo = ['softmax_cross_entropy'],
            me = ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = gpus, mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )
    
    # Load .eddl file if it is the case
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net



def recurrent_GRU(input_shape, num_classes, filename = None, gpus = [1]):

    if filename is not None and filename.endswith('.onnx'):
        # Load .onnx file if it is the case

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:
        # Create the model from scratch

        in_ = eddl.Input(input_shape)
        #
        layer = eddl.GRU(in_, 256)
        layer = eddl.Flatten(layer)
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine = True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    
    #print(f'{initialize=}')
    eddl.build(
            net,
            o = eddl.adam(lr = 1.0e-4),
            lo = ['softmax_cross_entropy'],
            me = ['categorical_accuracy'],
            cs = eddl.CS_GPU(g = gpus, mem = 'full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights = initialize
            )
    
    # Load .eddl file if it is the case
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    net = recurrent_GRU(input_shape = [64, 19, 256], num_classes = 2)
