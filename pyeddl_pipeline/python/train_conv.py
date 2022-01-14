"""
    Script for training convolutional neural network models to perform
    the detection of Epilepsy Seizures on a EEG signal. This task is part of the
    Use Case 13 of DeepHealth project. 

    This script uses Keras toolkit to create and train the neural networks.

    Authors:
        DeepHealth team @ PRHLT, UPV
"""

import os
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy

from data_utils_detection import RawDataGenerator2
from models import create_model

from pyeddl import eddl
from pyeddl.tensor import Tensor
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score


def main(args):

    index_training = [args.index]
    index_validation = [args.index_val]
    patient_id = args.id
    model_id = args.model    
    epochs = args.epochs
    batch_size = args.batch_size
    resume_dir = args.resume
    starting_epoch = args.starting_epoch
    gpus = args.gpus
    initial_lr = args.lr
    optimizer = args.opt

    model_checkpoint = None

    # Create dirs for the experiment
    if resume_dir is None:
        os.makedirs('experiments', exist_ok=True)
        exp_name = f'detection_conv_{patient_id}_{model_id}_{optimizer}_{initial_lr}'
        exp_dir = f'experiments/{exp_name}'

        exp_time = datetime.now().strftime("%d-%b_%H:%M")
        exp_dir = f'{exp_dir}_{exp_time}'

        os.makedirs(exp_dir, exist_ok=False)
        os.makedirs(exp_dir + '/models')
    else:
        exp_dir = resume_dir
        model_dir = exp_dir + '/models'
        for f in os.listdir(model_dir):
            if 'last' in f:
                model_checkpoint = os.path.join(model_dir, f)
        #
        if model_checkpoint is None:
            raise Exception(f'Last model not found in {model_dir}')
        #


    # Create data generator objects

    # Data Generator Object for training
    print('\n\nCreating Training Data Generator...', file=sys.stderr)
    dg = RawDataGenerator2(index_filenames=index_training,
                    window_length = args.window_length, # in seconds
                    shift = args.shift, # in seconds
                    sampling_rate = 256, # in Hz
                    batch_size=batch_size,
                    do_standard_scaling=True,
                    in_training_mode=True,
                    balance_batches=True,
                    patient_id=patient_id)


    print('\n\nCreating Validation Data Generator...', file=sys.stderr)
    dg_val = RawDataGenerator2(index_filenames=index_validation,
                    window_length = args.window_length, # in seconds
                    shift = args.shift, # in seconds
                    sampling_rate = 256, # in Hz
                    batch_size=batch_size,
                    do_standard_scaling=True,
                    in_training_mode=False,
                    balance_batches=False,
                    patient_id=patient_id)

    
    # Get input shape
    x, y = dg[0]
    #print(x.shape)
    input_shape = x.shape[1:]
    print(input_shape)


    log_file = open(f'{exp_dir}/training.log', 'w')
    log_file.write('epoch, train_acc, train_loss, val_acc,'
                  + ' val_f1score, val_balanced_acc\n')
    log_file.flush()


    net = create_model(model_id,
                       input_shape,
                       num_classes=1,
                       filename=model_checkpoint,
                       lr=initial_lr,
                       opt=optimizer,
                       gpus=gpus)
    #

    best_val_score = 0.0

    for epoch in range(starting_epoch, epochs):

        print(f'\nTraining epoch {epoch+1} of {epochs}...', file=sys.stderr)

        dg.shuffle_data()

        # TRAINING STAGE
        eddl.reset_loss(net)

        # Set a progress bar for the training loop
        pbar = tqdm(range(len(dg)))

        for i in pbar:
            # Load batch of data
            x, y = dg[i]

            # Add gaussian noise
            if model_id == 'conv1':
                noise = numpy.random.standard_normal(x.shape) * 0.25
                x += noise

            x = Tensor.fromarray(x) # (batch_size, 1, 2560, 23)
            y = Tensor.fromarray(y) # (2, 1)

            # Forward and backward of the channel through the net
            eddl.train_batch(net, [x], [y])

            losses = eddl.get_losses(net)
            #metrics = eddl.get_metrics(net)

            pbar.set_description(f'Training[loss={losses[0]:.5f}, acc=Not Available]')

        print()

        training_loss = losses[0]
        
        # VALIDATION
        print(f'\nValidation epoch {epoch+1}', file=sys.stderr)

        Y_true = list()
        Y_pred = list()

        for j in tqdm(range(len(dg_val))):
            x, y = dg_val[j]
            
            x = Tensor.fromarray(x)
            # Forward and backward of the channel through the net
            (y_pred, ) = eddl.predict(net, [x])

            y_pred = y_pred.getdata()
            y_pred = y_pred.ravel()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0


            Y_pred += y_pred.astype(int).tolist()
            Y_true += y.ravel().astype(int).tolist()
        #
        
        y_true = numpy.array(Y_true) * 1.0
        y_pred = numpy.array(Y_pred) * 1.0

        # Calculate other metrics
        val_accuracy = sum(y_true == y_pred) / len(y_true)
        cnf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        fscore = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        
        print('***************************************************************\n', file=sys.stderr)
        print(f'Epoch {epoch + 1}: Validation results\n', file=sys.stderr)
        print(' -- Single channel results (no combination of channels) --\n', file=sys.stderr)
        print(f'Validation acc : {val_accuracy * 100.0:.2f}', file=sys.stderr)
        print(f'Validation macro f1-score : {fscore:.4f}', file=sys.stderr)
        print(f'Validation balanced acc : {balanced_acc * 100.0:.2f}', file=sys.stderr)
        print('Confussion matrix:', file=sys.stderr)
        print(f'{cnf_matrix}\n', file=sys.stderr)
        print('Classification report:', file=sys.stderr)
        print(report, file=sys.stderr)

        print('\n--------------------------------------------------------------\n', file=sys.stderr)


        log_file.write('%d,%g,%g,%g,%g,%g\n' % (epoch, -1, training_loss,
            val_accuracy, fscore, balanced_acc))

        log_file.flush()

        # Save best model
        if (balanced_acc > best_val_score):
            best_val_score = balanced_acc
            eddl.save_net_to_onnx_file(net, f'{exp_dir}/models/best_model_epoch_{epoch:04d}_val_acc_{balanced_acc:.4f}.onnx')
            #eddl.save(net, f'{exp_dir}/models/best_model_epoch_{epoch:04d}_val_acc_{balanced_acc:.4f}.eddl')
        
        eddl.save_net_to_onnx_file(net, f'{exp_dir}/models/last.onnx')
        #eddl.save(net, f'{exp_dir}/models/last.eddl')

            

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser(description='Script for training models' + 
        ' to detect Epilepsy Seizures.')

    general_args = parser.add_argument_group("General Arguments")

    dg_args = parser.add_argument_group("Data Loader Arguments")

    resume_args = parser.add_argument_group("Arguments to resume an experiment (Optional)")

    general_args.add_argument('--index', help='Index of recordings to use for training. ' + 
                        'Example: "../indexes_detection/chb01/train.txt"')

    general_args.add_argument('--index-val', help='Index of recordings to use for validation. ' + 
                        'Example: "../indexes_detection/chb01/validation.txt"')
    
    general_args.add_argument('--id', help='Id of the patient, e.g. "chb01".', required=True)

    general_args.add_argument('--model', help='Model id to use: "conv1".',
                         default='conv1')

    general_args.add_argument('--epochs', type=int, help='Number of epochs to' +
         ' perform. Default -> 10', default=1)
    
    general_args.add_argument('--batch-size', type=int, help='Batch size. Default -> 20',
        default=20)

    general_args.add_argument('--lr', type=float, help='Initial learning rate. Default -> 0.0001',
        default=0.0001)

    general_args.add_argument('--opt', help='Optimizer: "adam", "sgd". Default -> adam',
        default='adam')

    general_args.add_argument("--gpus", help='Sets the number of GPUs to use.'+ 
        ' Usage "--gpus 1 1" (two GPUs)', nargs="+", default=[1], type=int)


    # Arguments of the data generator
    dg_args.add_argument('--window-length', type=float, help='Window length '
    + ' in seconds. Default -> 1', default=10)

    dg_args.add_argument('--shift', type=float, help='Window shift '
    + ' in seconds. Default -> 0.5', default=0.25)


    # Arguments to resume an experiment
    resume_args.add_argument('--starting-epoch', help='Number of the epoch to start ' + 
                        'the training again. (--epochs must be the total ' +
                        'number of epochs to be done, including the epochs ' +
                        'already done before resuming)', type=int, default=0)
     
    resume_args.add_argument('--resume', help='Directory of the experiment dir to resume. (optional)',
                default=None)

    main(parser.parse_args())