"""
    Script for testing recurrent neural network models to perform
    the detection of Epilepsy Seizures on a EEG signal. This task is part of the
    Use Case 13 of DeepHealth project. 

    This script uses pyEDDL library to test the neural network models.

    Authors:
        DeepHealth team @ PRHLT, UPV
"""

import os
import sys
import argparse
from tqdm import tqdm
import numpy
from models import create_model
from data_utils_detection import RawRecurrentDataGenerator
from pyeddl import eddl
from pyeddl.tensor import Tensor
from sklearn.metrics import f1_score, confusion_matrix, classification_report



def calculate_detection_metrics(y_true,
                                y_pred,
                                sample_shift = 0.5, # in seconds
                                analysis_window_size = 20, # in time steps
                                ratio_for_positive_alarm = 0.8,
                                detection_threshold = 10 # in seconds
                                ):
    """
        This function calculates metrics for the detection of seizures
        such as Accuracy of the detection alarms, Sensitivity, Specifity and
        False Positive Ratio per hour.

        Parameters
        ------------------------------------------------------------------

        :param numpy.array y_true:
            Numpy array with the ground truth.

        :param numpy.array y_pred:
            Numpy array with the predictions.

        :param float sample_shift:
            The number of seconds that represent a time step.
            If betweeen the samples you shift half second, you have two 
            predictions each second. Therefore, sample_shift will be 0.5

        :param int analyis_window_size:
            Number of time steps to analyze each time to decide the raising
            of an alarm or not.

        :param float ratio_for_positive_alarm:
            Percentage of the analysis window to be predicted as positive to 
            consider raising an alarm. 

        :param int detection_threshold:
            Number of seconds within the model has to detect each seizure.


        :returns tuple metrics:
            Tuple with the calculated metrics.

    """
    
    assert len(y_true) == len(y_pred)

    #interictal_time = len(y_true[y_true == 0]) * sample_shift  / 3600 # in hours

    # Results at alarm level
    num_seizures = 0
    false_positives = 0
    true_positives = 0
    latencies = list()

    # Results at sliding window level
    y_true_window = list()
    y_pred_window = list()


    i = analysis_window_size # To iterate over the results

    current_state = y_true[i-1] # 0 for interictal, 1 for ictal
    new_seizure = False   # Flag to know if we detected a seizure for the first time

    # Iterate over y_pred
    while i < len(y_pred):
        
        # Check if there is a new seizure and count it
        if y_true[i-1] == 1 and y_true[i-2] == 0:
            # New seizure
            num_seizures += 1


        # Get number of positive predictions inside the analysis window
        positive_predictions = sum(y_pred[i-analysis_window_size : i])
        positive_ratio = positive_predictions / analysis_window_size


        if positive_ratio >= ratio_for_positive_alarm:
            # Raise an alarm, possible seizure detected

            # Add sliding window results
            y_pred_window.append(1)
            y_true_window.append(y_true[i-1])

            # Check if we are detecting a seizure for the first time
            if current_state == 0:
                # Detected the seizure for the first time
                current_state = 1
                new_seizure = True # To store latency in case of correct detection

            if y_true[i-1] == 0:
                # False alarm
                false_positives += 1

            elif y_true[i-1] == 1:
                # Get distance to the onset
                # Calculate latency - find the onset of the seizure
                j = i - 1
                while y_true[j] == 1:
                    j -= 1
                #

                latency = (i - 1 - j) * sample_shift   # (i - 1) - (j + 1) + 1

                if latency <= detection_threshold:
                    # True positive
                    true_positives += 1

                    # Only save latency in case of the first time we detect the seizure
                    if new_seizure:
                        latencies.append(latency)
                else:
                    # False positive - too late
                    # TODO: Consider if not counting as false positive
                    false_positives += 1

            else:
                raise Exception(f'Unexpected value for ground truth {y_true[i-1]}')
            #
        else:
            # Negative prediction

            # Add sliding window results
            y_pred_window.append(0)
            y_true_window.append(y_true[i-1])

            # Check if we are detecting a normal state for the first time
            if current_state == 1:
                # Detected the normal state for the first time
                current_state = 0
        #
        # Update i
        i += 1
    #

    # Results at sliding window level
    y_true_window = numpy.array(y_true_window) * 1.0
    y_pred_window = numpy.array(y_pred_window) * 1.0
    accuracy_window = sum(y_true_window == y_pred_window) / len(y_true_window)


    # Results at alarm level
    average_latency = sum(latencies) / len(latencies)
    recall = len(latencies) / num_seizures  # Recall
    false_positives_per_hour = false_positives / (len(y_pred) * sample_shift)

    return accuracy_window, average_latency, false_positives_per_hour, recall





def main(args):
    
    """
        Test a model of a patient in the detection of seizures.
    """

    # Arguments
    index_test = [args.index]
    patient_id = args.id
    model_id = args.model
    batch_size = args.batch_size
    gpus = args.gpus
    exp_dir = args.dir
    #model_filename = args.model_filename

    # Create Data Generator object for the test set
    print('Creating Test Data Generator...')
    dg_test = RawRecurrentDataGenerator(index_filenames=index_test,
                          window_length = 1,
                          shift = 0.5, 
                          timesteps = 19,
                          sampling_rate = 256, # in Hz
                          batch_size=batch_size,
                          in_training_mode=False,
                          patient_id=patient_id)
    #

    model_dir = os.path.join(exp_dir, 'models')

    # Find best model in the models directory
    best_model = dict()  # {epoch: model_filename}
    for file in os.listdir(model_dir):
        if 'best' in file:
            w = file.split('_')
            for i in range(len(w)):
                if w[i] == 'epoch':
                    epoch = int(w[i + 1])
                    break
            #
            best_model[epoch] = file
    #
    # Get the highest epoch model filename - which is the best model -
    best_model_name = best_model[max(best_model.keys())]

    print(f'Evaluating best model with the test set -> {best_model_name}')

    model_filename = os.path.join(model_dir, best_model_name)

    # Load the model in the eddl
    print('Loading the model...')
    net = create_model(model_id=model_id,
                       input_shape=None, # Not needed if we are loading
                       num_classes=2,
                       filename=model_filename,
                       gpus=gpus)
    #


    # Get predictions for the test set with the best model
    print('Testing the model with the test signals...')
    Y_true_single_channel = list()
    Y_pred_single_channel = list()
    Y_true = list()
    Y_pred = list()

    for j in tqdm(range(len(dg_test))):
    #for j in tqdm(range(700, 894)):
        x, y = dg_test[j]
        #print(x.shape, y.shape)
        
        channels_y_pred = list()
        for channel in range(x.shape[3]):
            x_channel = x[:, :, :, channel]
            #print(x_channel.shape)
            channel_tensor_batch = Tensor.fromarray(x_channel)
            # Forward and backward of the channel through the net
            (y_pred, ) = eddl.predict(net, [channel_tensor_batch])

            y_pred = y_pred.getdata()
            
            #print(y_pred.getdata().shape)
            channels_y_pred.append(y_pred)
            Y_pred_single_channel += y_pred.argmax(axis=1).tolist()
            Y_true_single_channel += y.tolist()
        
        channels_y_pred = numpy.array(channels_y_pred)
        # (23, batch_size, 2)
        channels_y_pred = numpy.sum(channels_y_pred, axis=0)
        # print(channels_y_pred.shape) -> (batch_size, 2)
        
        Y_true += y.tolist()
        Y_pred += channels_y_pred.argmax(axis=1).tolist()
    #

    y_true = numpy.array(Y_true) * 1.0
    y_pred = numpy.array(Y_pred) * 1.0
    y_true_single_channel = numpy.array(Y_true_single_channel) * 1.0
    y_pred_single_channel = numpy.array(Y_pred_single_channel) * 1.0

    
    # Calculate and print basic metrics

    test_accuracy_single_channel = sum(y_true_single_channel == y_pred_single_channel) / len(y_true_single_channel)
    cnf_matrix = confusion_matrix(y_true_single_channel, y_pred_single_channel)
    report = classification_report(y_true_single_channel, y_pred_single_channel)
    fscore_single_channel = f1_score(y_true_single_channel, y_pred_single_channel, labels=[0, 1], average='macro')
    
    print('***************************************************************\n', file=sys.stderr)
    print(f'Test results\n', file=sys.stderr)
    print(' -- Single channel results (no combination of channels) --\n', file=sys.stderr)
    print(f'Test accuracy : {test_accuracy_single_channel}', file=sys.stderr)
    print(f'Test macro f1-score : {fscore_single_channel}', file=sys.stderr)
    print('Confussion matrix:', file=sys.stderr)
    print(f'{cnf_matrix}\n', file=sys.stderr)
    print('Classification report:')
    print(report, file=sys.stderr)

    print('\n--------------------------------------------------------------\n', file=sys.stderr)

    test_accuracy = sum(y_true == y_pred) / len(y_true)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    fscore = f1_score(y_true, y_pred, labels=[0, 1], average='macro')

    print(' -- All channels involved (combined for each timestamp) --\n', file=sys.stderr)
    print(f'Test accuracy : {test_accuracy}', file=sys.stderr)
    print(f'Test macro f1-score : {fscore}', file=sys.stderr)
    print('Confussion matrix:', file=sys.stderr)
    print(f'{cnf_matrix}\n', file=sys.stderr)
    print('Classification report:')
    print(report, file=sys.stderr)
    
    print('\n--------------------------------------------------------------\n', file=sys.stderr)
    

    # Calculate and print other metrics: 
    acc_window, latency, fp_h, recall = calculate_detection_metrics(y_true, y_pred)

    print('Global metrics after inference\n\n', file=sys.stderr)
    print(f'Accuracy of the sliding window: {acc_window * 100.0:.4f}', file=sys.stderr)
    print(f'Percentage of detected seizures: {recall * 100.0:.4f}', file=sys.stderr)
    print(f'Average latency: {latency} seconds', file=sys.stderr)
    print(f'False Alarms per Hour: {fp_h}', file=sys.stderr)

    print('***************************************************************\n\n', file=sys.stderr)



# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser(description='Script for training recurrent models' + 
        ' to detect epilepsy on UC13. \nThis script loads the best model '
        + 'saved in the experiments directory specified and performs the inference, '
        + 'returning the obtained metrics.', 
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--index', help='Index filename to use for test.',
                        required=True)

    parser.add_argument('--id', help='Id of the patient.', required=True)

    parser.add_argument('--model', help='Model identifier. "lstm" "gru"',
                        required=True)

    parser.add_argument('--dir', help='Directory of the experiment dir to test.'
                + ' Example: experiments/detection_recurrent_chb01_LSTM/',
                required=True)

    parser.add_argument('--batch-size', type=int, help='Batch size.',
        default=10)

    parser.add_argument("--gpus", help='Sets the number of GPUs to use.'+ 
        ' Usage "--gpus 1 1" (two GPUs)', nargs="+", default=[1], type=int)

    #parser.add_argument('--model-filename', help='Path to the model file to'
    #            + ' load. This is just needed if you want to test a specific'
    #            + ' model different than the best model of the experiment. '
    #            + 'If --dir is specified, the best model inside the'
    #            + ' models directory will be loaded automatically.',
    #            default=None)

    main(parser.parse_args())
