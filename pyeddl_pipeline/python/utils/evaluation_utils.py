"""
This script contains some functions to evaluate metrics on the Use Case 13
"""

def calculate_fpr(y_pred, k = 24, n = 30, sph = 60, sop = 1800, window_length = 10):
    """
    Performs the FPR (False Positive Rate) evaluation of a signal.

    Parameters:

        :param numpy.array y_pred:
            Prediction array
        
        :param numpy.array y_true:
            Array with the true labels

        :param int k:
            Number of 'preictal' predictions needed in the last n
            predictions to consider an alarm.

        :param int n:
            Number of last predictions to use to decide if the system
            is rising an alarm.
        
        :param int sph:
            SPH (Seizure Prediction Horizon). It is the time between
            the alarm of a seizure and the beginning of the SOP.
            It is defined in seconds.

        :param int sop:
            SOP (Seizure Occurrence Period). It is the time when
            a seizure is expected to occur. It is defined in seconds.

        :param int window_length:
            Window length used to process the signal (in seconds).
        
    :return int
        The value of FPR per hour.

    """

    # Convert sph and sop to number of positions in the array
    sph = sph // window_length
    sop = sop // window_length

    # We will need the preictal period length because we only want
    # one alarm for each seizure. When an alarm is set, we will
    # skip the prediction values, so we start again calculating
    # the fpr after a preictal period length. Therefore, we will only
    # have one alarm per seizure. (Deprecated)

    # We will need preictal period length. It will be one hour in general.
    # If preictal is modified when processing data, it must be updated here!
    preictal_period = 3600 // window_length

    i = n
    true_predictions = 0
    false_predictions = 0
    while i < len(y_pred):
    #for i in range(n, len(y_pred)):
        # Get how many positive predictions are in the last n predictions
        positive_predictions = sum(y_pred[i - n: i])

        if positive_predictions >= k:
            # We have an alarm
            # See if in SPH there is not a seizure
            sop_start = i + sph
            if sop_start < len(y_pred):
                if (sop_start + sop) >= len(y_pred):
                    # True prediction, because the seizure starts at the end of the array
                    true_predictions += 1
                    break
                else:
                    # False prediction
                    false_predictions += 1
                    # Continue after SPH + SOP
                    i = i + sph + sop
            #
            else:
                # False prediction, seizure is starting in SPH period
                false_predictions += 1
                break
            #
        else:
            # No alarm, update i
            i += 1
    #
    #
    # False prediction rate = Number of false predictions / interictal data duration
    print(false_predictions, true_predictions)
    interictal_duration = (len(y_pred) - preictal_period) * window_length / 3600
    # FPR/h
    fpr = false_predictions / interictal_duration
    #
    return fpr