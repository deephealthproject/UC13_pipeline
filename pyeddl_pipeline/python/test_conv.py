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
from data_utils_detection import RawDataGenerator2
from pyeddl import eddl
from pyeddl.tensor import Tensor
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score



def calculate_detection_metrics(y_true,
                                y_pred,
                                sample_shift = 0.5, # in seconds
                                sliding_window_length = 20, # in time steps
                                alpha_pos = 0.2,
                                alpha_neg = 0.2,
                                detection_threshold = 20 # in seconds
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

        :param int sliding_window_length:
            Number of time steps to analyze each time to decide the raising
            of an alarm or not.

        :param float alpha_pos:
            Minimum percentage of the analysis window to be predicted as 
            positive to trigger a transition between normal state to ictal state.

        :param float alpha_neg:
            Maximum percentage of the analysis window to be predicted as 
            negative to trigger a transition between ictal state to normal state.

        :param int detection_threshold:
            Number of seconds within the model has to detect each seizure.


        :returns tuple metrics:
            Tuple with the calculated metrics.

    """
    
    assert len(y_true) == len(y_pred)


    # Results at alarm level
    num_seizures = 0
    false_positives = 0
    true_positives = 0
    latencies = list()

    # Results at sliding window level
    y_true_window = list()
    y_pred_window = list()


    current_state = y_pred[sliding_window_length-1] # 0 for normal, 1 for ictal
    predicted = False # To only predict each seizure once

    # Iterate over y_pred
    for i in range(sliding_window_length, len(y_pred)):

        # Check if there is a new seizure and count it
        if y_true[i-1] == 1 and y_true[i-2] == 0:
            # New seizure
            num_seizures += 1
            predicted = False # To only predict each seizure once


        # Get number of positive predictions inside the analysis window
        positive_predictions = sum(y_pred[i-sliding_window_length : i])
        positive_ratio = positive_predictions / sliding_window_length

        if positive_ratio >= alpha_pos and current_state == 0:
            # Trigger transition from normal to ictal
            current_state = 1

            # Raise an alarm, possible seizure detected


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
                #print(latency)

                if latency <= detection_threshold:
                    if not predicted:
                        # True positive
                        true_positives += 1
                        latencies.append(latency)
                        predicted = True

                else:
                    # False positive - too late
                    false_positives += 1

            else:
                raise Exception(f'Unexpected value for ground truth {y_true[i-1]}')
            #

        elif positive_ratio <= alpha_neg and current_state == 1:
            # Trigger transition from ictal to normal
            current_state = 0

            # Add sliding window results
            y_pred_window.append(0)
            y_true_window.append(y_true[i-1])

        #
        # Add sliding window results
        y_pred_window.append(current_state)
        y_true_window.append(y_true[i-1])

        #print(positive_ratio, current_state, y_pred[i-1], y_true[i-1])
    #

    print(f'Num seizures: {num_seizures}', file=sys.stderr)

    # Results at sliding window level
    y_true_window = numpy.array(y_true_window) * 1.0
    y_pred_window = numpy.array(y_pred_window) * 1.0
    accuracy_window = sum(y_true_window == y_pred_window) / len(y_true_window)


    # Results at alarm level
    if len(latencies) == 0:
        average_latency = -1
        recall = 0
    else:
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

    # Create Data Generator object for the test set
    print('Creating Test Data Generator...', file=sys.stderr)

    dg_test = RawDataGenerator2(index_filenames=index_test,
                    window_length=args.window_length, # in seconds
                    shift=args.shift, # in seconds
                    sampling_rate=256, # in Hz
                    batch_size=batch_size,
                    do_standard_scaling=True,
                    in_training_mode=False,
                    balance_batches=False,
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

    print(f'Evaluating best model with the test set -> {best_model_name}', file=sys.stderr)

    model_filename = os.path.join(model_dir, best_model_name)

    # Load the model in the eddl
    print('Loading the model...', file=sys.stderr)
    net = create_model(model_id=model_id,
                       input_shape=None, # Not needed if we are loading a ONNX
                       num_classes=1,
                       filename=model_filename,
                       gpus=gpus)
    #


    # Get predictions for the test set with the best model
    print('Testing the model with the test signals...', file=sys.stderr)
    Y_true = list()
    Y_pred = list()

    for j in tqdm(range(len(dg_test))):
            x, y = dg_test[j]
            
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
    
    # Calculate and print basic metrics

    print('\n--------------------------------------------------------------\n', file=sys.stderr)

    test_accuracy = sum(y_true == y_pred) / len(y_true)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    fscore = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f' -- Patient {patient_id} test results --')
    print(f'Model:  {best_model_name}\n', file=sys.stderr)
    print(f'Test accuracy : {test_accuracy}', file=sys.stderr)
    print(f'Test macro f1-score : {fscore}', file=sys.stderr)
    print(f'Test balanced acc : {balanced_acc}', file=sys.stderr)
    print('Confussion matrix:', file=sys.stderr)
    print(f'{cnf_matrix}\n', file=sys.stderr)
    print('Classification report:', file=sys.stderr)
    print(report, file=sys.stderr)
    
    print('\n--------------------------------------------------------------\n', file=sys.stderr)

    # Calculate and print other metrics: 
    acc_window, latency, fp_h, recall = calculate_detection_metrics(
                                        y_true,
                                        y_pred,
                                        sample_shift=args.shift,
                                        sliding_window_length=args.inference_window,
                                        alpha_pos=args.alpha_pos,
                                        alpha_neg=args.alpha_neg,
                                        detection_threshold=args.detection_threshold
                                        )

    print('Global metrics after inference\n\n', file=sys.stderr)
    print(f'Accuracy of the sliding window: {acc_window * 100.0:.4f}', file=sys.stderr)
    print(f'Percentage of detected seizures: {recall * 100.0:.4f}', file=sys.stderr)
    print(f'Average latency: {latency} seconds', file=sys.stderr)
    print(f'False Alarms per Hour: {fp_h}', file=sys.stderr)

    print('***************************************************************\n\n', file=sys.stderr)



# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser(description='Script for testing CNN models' + 
        ' to detect epilepsy on UC13. \nThis script loads the best model '
        + 'saved in the experiments directory specified and performs the inference, '
        + 'returning the obtained metrics.', 
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--index', help='Index filename to use for test.',
                        required=True)

    parser.add_argument('--id', help='Id of the patient.', required=True)

    parser.add_argument('--model', help='Model identifier. "conv1"',
                        required=True)

    parser.add_argument('--dir', help='Directory of the experiment to test.'
                + ' Example: experiments/detection_conv1_chb01/',
                required=True)

    parser.add_argument('--batch-size', type=int, help='Batch size. Default -> 20',
        default=20)

    parser.add_argument("--gpus", help='Sets the number of GPUs to use.'+ 
        ' Usage "--gpus 1 1" (two GPUs)', nargs="+", default=[1], type=int)

    # Arguments of the data generator
    parser.add_argument('--window-length', type=float, help='Window length '
    + ' in seconds. Default -> 1', default=1)

    parser.add_argument('--shift', type=float, help='Window shift '
    + ' in seconds. Default -> 0.5', default=0.5)


    # Args for the alarm function
    parser.add_argument('--inference-window', type=int, help='Length of the '
        + 'sliding window to use after inferencing with the CNN. Default -> 20',
        default=20)

    parser.add_argument('--alpha-pos', type=float, help='Minimum rate of'
        + ' positive predicted samples in the sliding window for triggering'
        + ' a transition between normal state to ictal state. Default -> 0.4',
        default=0.4)
    
    parser.add_argument('--alpha-neg', type=float, help='Maximum rate of'
        + ' positive predicted samples in the sliding window for triggering'
        + ' a transition between normal state to ictal state. Default -> 0.4',
        default=0.4)

    parser.add_argument('--detection-threshold', type=int, help='Number of '
        + 'seconds from the seizure onset to allow the detection. Default -> 20',
        default=20)

    main(parser.parse_args())