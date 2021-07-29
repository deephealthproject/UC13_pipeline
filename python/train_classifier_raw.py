import os
import sys
import numpy

from pyeddl import eddl
from pyeddl.tensor import Tensor

from data_utils import DataGenerator, RawDataGenerator
#from data_utils_eeg import PairDataGenerator
from models_01 import model_classifier_1a, model_classifier_2a
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


if __name__ == '__main__':

    starting_epoch = 0
    epochs = 10
    batch_size = 100
    model_filename = None
    model_id = '1a'
    #index_filenames = list()
    data_dir = None
    early_stopping_epochs = 5
    patient_id = None
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
        elif sys.argv[i] == '--id':
            patient_id = sys.argv[i + 1]
    
    dg = RawDataGenerator(batch_size = batch_size,
                        window_length = 256 * 10,
                        shift = 256 * 10,
                        min_interictal_length = 256 * 3600 * 4, # 4 hours
                        preictal_length = 256 * 3600, # 1 hour
                        do_shuffle = True,
                        load_stats = False,
                        mode = 'train',
                        patient_id = patient_id)

    print(f"Number of seizures: %d" % dg.num_seizures)

    x, y = dg[0]
    print(x.shape)
    input_shape = (1,) + x.shape[1:]
    #if model_id == '1a':
    #    net = model_classifier_1a(input_shape, num_classes = 2, filename = model_filename)
    #elif model_id == '2a':
    #    net = model_classifier_2a(input_shape, num_classes = 2, filename = model_filename)
    ##elif model_id == '2':
    ##    net = model_2(input_shape, input_shape, filename = model_filename)
    #else:
    #    raise Exception('You have to indicate a model id!')


    log_file = open(f'log/model_classifier_{model_id}_patient_{patient_id}_train.log', 'a')
    log_file.write("fold,epoch,bce,train_acc,val_acc,val_precision_preictal,val_recall_preictal,val_fscore_preictal\n")
    log_file.flush()

    test_log = open(f'log/model_classifier_{model_id}_patient_{patient_id}_test.log', 'a')
    test_log.write("fold,test_acc,test_precision_preictal,test_recall_preictal,test_fscore_preictal\n")
    test_log.flush()

    for fold in range(dg.num_seizures):
        #
        if model_id == '1a':
            net = model_classifier_1a(input_shape, num_classes = 2, filename = model_filename)
        elif model_id == '2a':
            net = model_classifier_2a(input_shape, num_classes = 2, filename = model_filename)
        else:
            raise Exception('You have to indicate a model id!')
        #
        score_to_use = 0 # Score to use for selecting the best model when training 
        best_scores = [0.0, 0.0, 0.0, 0.0] # Accuracy, Precision(preictal), Recall(preictal), F1-score(preictal)
        patience = 0 # Number of epochs without improving best score
        best_model = net # Best model
        for epoch in range(starting_epoch, epochs):
            print()
            dg.mode = 'train'
            print('fold', fold, 'epoch:', epoch, 'num batches:', len(dg))
            eddl.reset_loss(net)
            
            # Training
            for j in range(len(dg)):
                x, y = dg[j]
                x = numpy.expand_dims(x, axis = 1)
                indices = list(range(len(x)))
                x = Tensor.fromarray(x)
                _y_ = numpy.zeros([len(y),2])
                _y_[y == 0, 0] = 1
                _y_[y == 1, 1] = 1
                y = Tensor.fromarray(_y_)

                eddl.train_batch(net, [x], [y], indices = indices)
                eddl.print_loss(net, j)
                print('\r', end = '')
            print()

            #log_file.write("fold %d    epoch %d   bce %g  acc %g\n" % (fold, epoch, eddl.get_losses(net)[0], eddl.get_metrics(net)[0]))
            #log_file.flush()
            #eddl.save_net_to_onnx_file(net, f'models/model_classifier_{model_id}-{epoch}.onnx')

            # Validation
            Y_true = list()
            Y_pred = list()
            dg.mode = 'val'
            for j in range(len(dg)):
                x, y_true = dg[j]
                x = numpy.expand_dims(x, axis = 1)
                indices = list(range(len(x)))
                x = Tensor.fromarray(x)
                (y_pred, ) = eddl.predict(net, [x])
                y_pred = y_pred.getdata()
                Y_true.append(y_true)
                Y_pred.append(y_pred.argmax(axis = 1))

            y_true = numpy.hstack(Y_true) * 1.0
            y_pred = numpy.hstack(Y_pred) * 1.0
            #print('fold %d validation accuracy epoch %d = %g' % ( fold, epoch, sum(y_true == y_pred) / len(y_true)))
            #log_file.write('fold %d validation accuracy epoch %d = %g\n' % ( fold, epoch, sum(y_true == y_pred) / len(y_true))) # eddl.get_metrics(net)[0]))
            val_accuracy = sum(y_true == y_pred) / len(y_true)
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)

            scores = [val_accuracy, precision[1], recall[1], fscore[1]]
            if scores[score_to_use] > best_scores[score_to_use]:
                best_scores = scores.copy()
                best_model = net # copy() 'pyeddl._core.Net' object has no attribute 'copy'
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_epochs:
                    print(f'Stopping training. Model not improving for {patience} consecutive epochs')
                    break


            print(confusion_matrix(y_true, y_pred, labels = [0, 1]))
            print(classification_report(y_true, y_pred, target_names = ['inter-ictal', 'pre-ictal']))
            log_file.write("%d,%d,%g,%g,%g,%g,%g,%g\n" % (fold, epoch, eddl.get_losses(net)[0], 
                                                eddl.get_metrics(net)[0], 
                                                val_accuracy,
                                                precision[1], recall[1],
                                                fscore[1]))
            log_file.flush()


            dg.on_epoch_end()
        # for epoch loop
        #
        # Test
        print(f"Testing the model at fold {fold}")
        Y_true = list()
        Y_pred = list()
        dg.mode = 'test'
        for j in range(len(dg)):
            x, y_true = dg[j]
            x = numpy.expand_dims(x, axis = 1)
            indices = list(range(len(x)))
            x = Tensor.fromarray(x)
            (y_pred, ) = eddl.predict(best_model, [x])
            y_pred = y_pred.getdata()
            Y_true.append(y_true)
            Y_pred.append(y_pred.argmax(axis = 1))

        y_true = numpy.hstack(Y_true) * 1.0
        y_pred = numpy.hstack(Y_pred) * 1.0
        #print('fold %d validation accuracy epoch %d = %g' % ( fold, epoch, sum(y_true == y_pred) / len(y_true)))
        #log_file.write('fold %d validation accuracy epoch %d = %g\n' % ( fold, epoch, sum(y_true == y_pred) / len(y_true))) # eddl.get_metrics(net)[0]))
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
        print(confusion_matrix(y_true, y_pred, labels = [0, 1]))
        print(classification_report(y_true, y_pred, target_names = ['inter-ictal', 'pre-ictal']))
        test_log.write("%d,%g,%g,%g,%g,%g\n" % (fold, sum(y_true == y_pred) / len(y_true),
                                            precision[1], recall[1],
                                            fscore[1]))
        test_log.flush()

    # for fold
    log_file.close()
    test_log.close()
