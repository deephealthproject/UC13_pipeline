"""
    Script for training recurrent neural network models to perform
    the detection of Epilepsy Seizures on a EEG signal. This task is part of the
    Use Case 13 of DeepHealth project. 

    This script uses pyEDDL library to create and train the neural networks.

    Authors:
        DeepHealth team @ PRHLT, UPV
"""


import os
import sys
import argparse
from datetime import datetime
import numpy
from tqdm import tqdm

from data_utils_detection import RawRecurrentDataGenerator
from models import create_model
from pyeddl import eddl
from pyeddl.tensor import Tensor
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score



def main(args):

    """
    Train a model on epilepsy detection with recurrent neural networks.
    """

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
        exp_name = f'detection_recurrent_{patient_id}_{model_id}_{optimizer}_{initial_lr}'
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


    
    log_file = open(f'{exp_dir}/training_log.txt', 'w' if resume_dir is None else 'a')
    log_file.write('epoch, train_acc, train_loss, val_acc_single_channel,'
        + ' val_f1score_single_channel, val_acc, val_f1score, val_balanced_acc\n')
    log_file.flush()

    # Data Generator Object for training
    print('\n\nCreating Training Data Generator...', file=sys.stderr)
    dg = RawRecurrentDataGenerator(index_filenames=index_training,
                          window_length=args.window_length, # in seconds
                          shift=args.shift, # in seconds
                          timesteps=args.timesteps, # in seconds
                          sampling_rate=256, # in Hz
                          batch_size=batch_size,
                          in_training_mode=True,
                          balance_batches=True,
                          patient_id=patient_id)
    #

    print('\n\nCreating Validation Data Generator...', file=sys.stderr)
    dg_val = RawRecurrentDataGenerator(index_filenames=index_validation,
                          window_length=args.window_length,
                          shift=args.shift, 
                          timesteps=args.timesteps,
                          sampling_rate=256, # in Hz
                          batch_size=batch_size,
                          in_training_mode=False,
                          patient_id=patient_id)

    
    print(dg.input_shape)

    net = create_model(model_id,
                       dg.input_shape,
                       num_classes=1,
                       filename=model_checkpoint,
                       lr=initial_lr,
                       opt=optimizer,
                       gpus=gpus)
    #

    best_val_acc = 0.0

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

            y = Tensor.fromarray(y.reshape((len(y), 1, 1)))

            for channel in range(x.shape[3]):
                x_channel = x[:, :, :, channel]
                
                channel_tensor_batch = Tensor.fromarray(x_channel)
                # Forward and backward of the channel through the net
                eddl.train_batch(net, [channel_tensor_batch], [y])

                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                pbar.set_description(f'Training[loss={losses[0]:.5f}, acc={metrics[0]:.5f}]')

        print()

        training_loss = losses[0]
        training_acc = metrics[0]
        
        # VALIDATION
        print(f'\nValidation epoch {epoch+1}', file=sys.stderr)

        Y_true_single_channel = list()
        Y_pred_single_channel = list()
        Y_true = list()
        Y_pred = list()


        for j in tqdm(range(len(dg_val))):
            x, y = dg_val[j]
            
            channels_y_pred = list()
            for channel in range(x.shape[3]):
                x_channel = x[:, :, :, channel]

                channel_tensor_batch = Tensor.fromarray(x_channel)
                # Forward and backward of the channel through the net
                (y_pred, ) = eddl.predict(net, [channel_tensor_batch])

                y_pred = y_pred.getdata()
                y_pred = y_pred.ravel()
                
                channels_y_pred.append(y_pred)

                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                
                Y_pred_single_channel += y_pred.astype(int).tolist()
                Y_true_single_channel += y.ravel().astype(int).tolist()

            channels_y_pred = numpy.array(channels_y_pred)
            # Shape -> (23, batch_size)
            channels_y_pred = numpy.sum(channels_y_pred, axis=0)
            channels_y_pred = channels_y_pred / 23.0
            # Shape -> (batch_size,)
            channels_y_pred[channels_y_pred >= 0.5] = 1
            channels_y_pred[channels_y_pred < 0.5] = 0
            
            Y_true += y.ravel().astype(int).tolist()
            Y_pred += channels_y_pred.astype(int).tolist()
        #

        y_true = numpy.array(Y_true) * 1.0
        y_pred = numpy.array(Y_pred) * 1.0
        y_true_single_channel = numpy.array(Y_true_single_channel) * 1.0
        y_pred_single_channel = numpy.array(Y_pred_single_channel) * 1.0

        val_accuracy_single_channel = sum(y_true_single_channel == y_pred_single_channel) / len(y_true_single_channel)
        cnf_matrix = confusion_matrix(y_true_single_channel, y_pred_single_channel)
        report = classification_report(y_true_single_channel, y_pred_single_channel)
        fscore_single_channel = f1_score(y_true_single_channel, y_pred_single_channel, labels=[0, 1], average='macro')
        balanced_acc_single_channel = balanced_accuracy_score(y_true_single_channel, y_pred_single_channel)
        
        print('***************************************************************\n', file=sys.stderr)
        print(f'Epoch {epoch + 1}: Validation results\n', file=sys.stderr)
        print(' -- Single channel results (no combination of channels) --\n', file=sys.stderr)
        print(f'Validation acc : {val_accuracy_single_channel * 100.0:.2f}', file=sys.stderr)
        print(f'Validation macro f1-score : {fscore_single_channel:.4f}', file=sys.stderr)
        print(f'Validation balanced accuracy: {balanced_acc_single_channel * 100.0:.2f}', file=sys.stderr)
        print('Confussion matrix:', file=sys.stderr)
        print(f'{cnf_matrix}\n', file=sys.stderr)
        print('Classification report:', file=sys.stderr)
        print(report, file=sys.stderr)

        print('\n--------------------------------------------------------------\n', file=sys.stderr)

        val_accuracy = sum(y_true == y_pred) / len(y_true)
        cnf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        fscore = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        print(' -- All channels involved (combined for each timestamp) --\n', file=sys.stderr)
        print(f'Validation acc : {val_accuracy * 100.0:.2f}', file=sys.stderr)
        print(f'Validation macro f1-score : {fscore:.4f}', file=sys.stderr)
        print(f'Validation balanced acc : {balanced_acc * 100.0:.2f}', file=sys.stderr)
        print('Confussion matrix:', file=sys.stderr)
        print(f'{cnf_matrix}\n', file=sys.stderr)
        print('Classification report:', file=sys.stderr)
        print(report, file=sys.stderr)
        print('***************************************************************\n\n', file=sys.stderr)

        log_file.write('%d,%g,%g,%g,%g,%g,%g,%g\n' % (epoch, training_acc, training_loss,
            val_accuracy_single_channel, fscore_single_channel,
            val_accuracy, fscore, balanced_acc))

        log_file.flush()

        # Save best model
        if (balanced_acc > best_val_acc):
            best_val_acc = balanced_acc
            eddl.save_net_to_onnx_file(net, f'{exp_dir}/models/best_model_epoch_{epoch:04d}_val_acc_balanced_{balanced_acc:.4f}.onnx')
        
        eddl.save_net_to_onnx_file(net, f'{exp_dir}/models/last.onnx')

    #
    log_file.close()



#-------------------------------------------------------------------------------

if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser(description='Script for training recurrent models' + 
        ' to detect epilepsy on UC13.',
        formatter_class=argparse.RawTextHelpFormatter)

    general_args = parser.add_argument_group("General Arguments")

    dg_args = parser.add_argument_group("Data Loader Arguments")

    resume_args = parser.add_argument_group("Arguments to resume an experiment (Optional)")


    general_args.add_argument('--index', help='Index filename to use.', required=True)

    general_args.add_argument('--index-val', help='Index filename to use for validation.', required=True)

    general_args.add_argument('--id', help='Id of the patient.', required=True)

    general_args.add_argument('--model', help='Model id to use: "lstm", "gru".',
                         default='lstm')

    general_args.add_argument('--epochs', type=int, help='Number of epochs to' +
         ' perform. Default -> 10', default=10)
    
    general_args.add_argument('--batch-size', type=int, help='Batch size. Default -> 64',
        default=64)

    general_args.add_argument("--gpus", help='Sets the number of GPUs to use.'+ 
        ' Usage "--gpus 1 1" (two GPUs)', nargs="+", default=[1], type=int)

    general_args.add_argument('--lr', type=float, help='Initial learning rate. Default -> 0.0001',
        default=0.0001)

    general_args.add_argument('--opt', help='Optimizer: "adam", "sgd". Default -> adam',
        default='adam')

    # Arguments of the data generator
    dg_args.add_argument('--window-length', type=float, help='Window length '
    + ' in seconds. Default -> 1', default=1)

    dg_args.add_argument('--shift', type=float, help='Window shift '
    + ' in seconds. Default -> 0.5', default=0.5)

    dg_args.add_argument('--timesteps', type=int, help='Timesteps to use as a '
    + ' sequence. Default -> 19', default=19)

    # Arguments to resume an experiment
    resume_args.add_argument('--resume', help='Directory of the experiment dir to resume.',
                default=None)

    resume_args.add_argument('--starting-epoch', help='Number of the epoch to start ' + 
                        'the training again. (--epochs must be the total ' +
                        'number of epochs to be done, including the epochs ' +
                        'already done before resuming)', type=int, default=0)

    main(parser.parse_args())
