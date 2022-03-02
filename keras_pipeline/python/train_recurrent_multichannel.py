"""
    Script for training recurrent neural network models to perform
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
import matplotlib.pyplot as plt

from data_utils_detection import RawRecurrentDataGenerator
from models_keras import create_model

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             f1_score,
                             balanced_accuracy_score)



def main(args):
    index_training = [args.index]
    index_validation = [args.index_val]
    patient_id = args.id
    model_id = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    initial_lr = args.lr
    opt = args.opt
    resume_dir = args.resume
    starting_epoch = args.starting_epoch

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # Create experiment directory or use an existing one to resume training

    if resume_dir is not None:
        # Resume training
        exp_dir = resume_dir
        model_dir = os.path.join(exp_dir, 'models')
        model_filename = None

        for f in os.listdir(model_dir):
            if 'last' in f:
                model_filename = os.path.join(model_dir, f)
        #
        if model_filename is None:
            raise Exception(f'Last model not found in {model_dir}')
        #
    else:
        # Create dir for the experiment
        os.makedirs('keras_experiments', exist_ok=True)
        
        exp_dir = os.path.join('keras_experiments' ,
                f'detection_recurrent_{patient_id}_{model_id}_{opt}_{initial_lr}')

        exp_time = datetime.now().strftime("%d-%b_%H:%M")
        exp_dir = f'{exp_dir}_{exp_time}'
        os.makedirs(exp_dir)

        ## Create dir to store models
        model_dir = os.path.join(exp_dir, 'models')
        os.makedirs(model_dir)
    #
    
    
    # Create data generator objects

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


    # Get input shape
    x, y = dg[0]
    #print(x.shape)
    input_shape = x.shape[1:-1]
    #print(input_shape)
    

    # Create or Load the model

    if resume_dir is None:
        # Create model and define optimizer
        model = create_model(model_id, input_shape, 2)

        if opt == 'adam':
            optimizer = Adam(learning_rate=initial_lr)
        elif opt == 'sgd':
            optimizer = SGD(learning_rate=initial_lr)
        else:
            raise Exception(f'Wrong optimizer name, check help with -h.')


        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['binary_accuracy']
                    )
    
    else:
        # Load model, already compiled and with the optimizer state preserved
        model = keras.models.load_model(model_filename)
    #

    model.summary()


    # Log files
    log_filename = f'{exp_dir}/training.log'
    logger = open(log_filename, 'a')
    logger.write('epoch, train_acc, train_loss, val_acc, ' 
                + 'val_loss, val_f1score, val_balanced_acc\n')
    
    logger.flush()


    best_val_score = 0.0

    # Train the model
    for epoch in range(starting_epoch, epochs):

        print(f'\nTraining epoch {epoch+1} of {epochs}...', file=sys.stderr)
        dg.shuffle_data()

        # Set a progress bar for the training loop
        pbar = tqdm(range(len(dg)))

        for i in pbar:
            # Load batch of data
            x, y = dg[i]

            inputs = [x[:,:,:,channel] for channel in range(x.shape[-1])]

            #print(inputs[0].shape) -> (batch_size, 256, 20)
                
            # Forward and backward of the channel through the net
            outputs = model.train_on_batch(inputs, y=y, reset_metrics=False)

            pbar.set_description(f'Training[loss={outputs[0]:.5f}, acc={outputs[1]:.5f}]')
        #

        # Store training results
        train_loss = outputs[0]
        train_acc = outputs[1]


        # Validation
        print(f'\nValidation epoch {epoch+1}...', file=sys.stderr)

        Y_true = list()
        Y_pred = list()

        accumulated_loss = 0.0

        for j in tqdm(range(len(dg_val))):
            x, y = dg_val[j]

            inputs = [x[:,:,:,channel] for channel in range(x.shape[-1])]

            # Forward and backward of the channel through the net
            y_pred = model.predict(inputs)

            accumulated_loss += keras.losses.BinaryCrossentropy()(y, y_pred)

            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            Y_pred += y_pred.ravel().astype(int).tolist()
            Y_true += y.ravel().astype(int).tolist()
        #

        y_true = numpy.array(Y_true) * 1.0
        y_pred = numpy.array(Y_pred) * 1.0

        print(len(y_true), len(y_pred), y_true.shape, y_pred.shape)
        # Calculate validation loss
        val_loss = accumulated_loss / len(dg_val)

        print('\n--------------------------------------------------------------\n', file=sys.stderr)

        val_accuracy = sum(y_true == y_pred) / len(y_true)
        print(val_accuracy)
        cnf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        fscore = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        print(f'Validation acc : {val_accuracy * 100.0:.2f}', file=sys.stderr)
        print(f'Validation macro f1-score : {fscore:.4f}', file=sys.stderr)
        print(f'Validation balanced acc : {balanced_acc * 100.0:.2f}', file=sys.stderr)
        print('Confussion matrix:', file=sys.stderr)
        print(f'{cnf_matrix}\n', file=sys.stderr)
        print('Classification report:', file=sys.stderr)
        print(report, file=sys.stderr)
        print('***************************************************************\n\n', file=sys.stderr)


        logger.write('%d,%g,%g,%g,%g,%g,%g\n' % (epoch, train_acc, train_loss,
            val_loss, val_accuracy, fscore, balanced_acc))

        logger.flush()


        if balanced_acc > best_val_score:
            # Save best model if score is improved
            best_val_score = balanced_acc
            model.save(f'{model_dir}/{model_id}_best_epoch' + f'_{epoch:04d}_{balanced_acc:.4f}.h5')
        
        # Save last model
        model.save(f'{model_dir}/{model_id}_last.h5')






# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser(description='Script for training models' + 
        ' to detect Epilepsy Seizures.')

    parser.add_argument('--index', help='Index of recordings to use for training. ' + 
                        'Example: "../indexes_detection/chb01/train.txt"')

    parser.add_argument('--index-val', help='Index of recordings to use for validation. ' + 
                        'Example: "../indexes_detection/chb01/validation.txt"')
    
    parser.add_argument('--id', help='Id of the patient, e.g. "chb01".', required=True)

    parser.add_argument('--model', help='Model id to use: "multi-lstm".',
                         default='multi-lstm')

    parser.add_argument('--epochs', type=int, help='Number of epochs to' +
         ' perform.', default=1)
    
    parser.add_argument('--batch-size', type=int, help='Batch size.',
        default=64)

    parser.add_argument('--lr', type=float, help='Initial learning rate. Default -> 0.0001',
        default=0.0001)

    parser.add_argument('--opt', help='Optimizer: "adam", "sgd". Default -> adam',
        default='adam')

    parser.add_argument('--gpu', help='Id of the gpu to use.'+ 
        ' Usage --gpu 0', default='0')


    # Arguments of the data generator
    parser.add_argument('--window-length', type=float, help='Window length '
    + ' in seconds. Default -> 1', default=1)

    parser.add_argument('--shift', type=float, help='Window shift '
    + ' in seconds. Default -> 0.5', default=0.5)

    parser.add_argument('--timesteps', type=int, help='Timesteps to use as a '
    + ' sequence. Default -> 19', default=19)



    # Arguments to resume an experiment
    parser.add_argument('--resume', help='Directory of the experiment dir to resume.',
                default=None)

    parser.add_argument('--starting-epoch', help='Number of the epoch to start ' + 
                        'the training again. (--epochs must be the total ' +
                        'number of epochs to be done, including the epochs ' +
                        'already done before resuming)', type=int, default=0)
     

    main(parser.parse_args())