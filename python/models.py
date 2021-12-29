from pyeddl import eddl


def create_model(model_id, 
                 input_shape,
                 num_classes, 
                 filename=None,
                 lr=0.0001,
                 opt='adam',
                 gpus=[1]):

    """
        Builds and returns a model given a model id,
        the number of classes to use, a filename to load
        the model (optional) and the gpus to use.
    """

    if model_id == 'lstm':
        net = recurrent_LSTM(input_shape,
                             num_classes=num_classes,
                             filename=filename,
                             lr=lr,
                             opt=opt,
                             gpus=gpus)
                             
    elif model_id == 'gru':
        net = recurrent_GRU(input_shape,
                            num_classes=num_classes,
                            filename=filename,
                            lr=lr,
                            opt=opt,
                            gpus=gpus)

    elif model_id == 'conv1':
        net = build_conv(input_shape,
                         num_classes=num_classes,
                         filename=filename,
                         lr=lr,
                         opt=opt,
                         gpus=gpus)

    else:
        raise Exception('You have to provide an existing model_id!')
    
    return net



def recurrent_LSTM(input_shape, num_classes, lr, opt, filename=None, gpus=[1]):

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
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine=True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    
    #print(f'{initialize=}')

    if opt == 'adam':
        optimizer = eddl.adam(lr=lr)
    elif opt == 'sgd':
        optimizer = eddl.sgd(lr=lr)
    else:
        raise Exception('Optimizer name not valid.')


    eddl.build(
            net,
            o=optimizer,
            lo=['softmax_cross_entropy'],
            me=['categorical_accuracy'],
            cs=eddl.CS_GPU(g=gpus, mem='full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights=initialize
            )
    
    # Load .eddl file if it is the case
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net



def recurrent_GRU(input_shape, num_classes, lr, opt, filename=None, gpus=[1]):

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
        layer = eddl.ReLu(eddl.BatchNormalization(eddl.Dense(layer,  256), affine=True))
        layer = eddl.Softmax(eddl.Dense(layer, num_classes))
        #
        out_ = layer

        net = eddl.Model([in_], [out_])
        initialize = True
    

    if opt == 'adam':
        optimizer = eddl.adam(lr=lr)
    elif opt == 'sgd':
        optimizer = eddl.sgd(lr=lr)
    else:
        raise Exception('Optimizer name not valid.')

    eddl.build(
            net,
            o=optimizer,
            lo=['softmax_cross_entropy'],
            me=['categorical_accuracy'],
            cs=eddl.CS_GPU(g=gpus, mem='full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights=initialize
            )
    
    # Load .eddl file if it is the case
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net




def build_conv(input_shape, num_classes, lr, opt, filename=None, gpus=[1]):

    if filename is not None and filename.endswith('.onnx'):
        # Load .onnx file if it is the case

        net = eddl.import_net_from_onnx_file(filename)
        initialize = False

    else:
        # Create the model from scratch

        in_ = eddl.Input(input_shape)
        #
        layer = eddl.L2(eddl.Conv2D(in_, 64, kernel_size=[128, 23], strides=[64, 1], padding='valid'), 0.00001)
        layer = eddl.Reshape(layer, (1, (input_shape[1] - 64) // 64, 64))

        for k in [128, 128, 256, 256]:
            layer = eddl.Conv2D(layer, k, kernel_size=[3, 3], strides=[1, 1], padding='same')
            layer = eddl.L2(layer, 0.00001)
            layer = eddl.ReLu(layer)
            layer = eddl.Dropout(layer, 0.25)
            layer = eddl.MaxPool2D(layer)

        layer = eddl.Conv2D(layer, 256, kernel_size=[3, 3], strides=[1, 1], padding='same')
        layer = eddl.Flatten(layer)

        layer = eddl.ReLu(eddl.L2(eddl.Dense(layer, 1024), 0.00001))
        layer = eddl.Dropout(layer, 0.5)

        layer = eddl.ReLu(eddl.L2(eddl.Dense(layer, 512), 0.00001))
        layer = eddl.Dropout(layer, 0.5)

        layer = eddl.L2(eddl.Dense(layer, 1), 0.00001)

        #
        out_ = eddl.Sigmoid(layer)

        net = eddl.Model([in_], [out_])
        initialize = True
    

    if opt == 'adam':
        optimizer = eddl.adam(lr=lr)
    elif opt == 'sgd':
        optimizer = eddl.sgd(lr=lr)
    else:
        raise Exception('Optimizer name not valid.')

    eddl.build(
            net,
            o=optimizer,
            lo=['binary_cross_entropy'],
            me=['binary_accuracy'],
            cs=eddl.CS_GPU(g=gpus, mem='full_mem'),
            #cs = eddl.CS_CPU(),
            init_weights=initialize
            )
    
    # Load .eddl file if it is the case
    if filename is not None and filename.endswith('.eddl'):
        eddl.load(net, filename)

    eddl.summary(net)
    
    return net



# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    net = recurrent_GRU(input_shape = [64, 19, 256], num_classes = 2)
