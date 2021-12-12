"""
    Neural Network models definition
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU


def create_model(model_id, input_shape, num_classes):
    """
        Base function for building different models.
    """

    if model_id == 'lstm':
        model = build_lstm(input_shape, num_classes)
    elif model_id == 'gru':
        model = build_gru(input_shape, num_classes)

    return model

# -----------------------------------------------------------------------------

def build_lstm(input_shape, num_classes):
    
    in_ = Input(shape=input_shape)
    
    layer = LSTM(256)(in_)
    layer = Flatten()(layer)
    layer = Dense(256, activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dense(num_classes, activation=None)(layer)
    
    out_ = Activation('softmax')(layer)

    model = Model(in_, out_)

    return model



def build_gru(input_shape, num_classes):
    
    in_ = Input(shape=input_shape)
    
    layer = GRU(256)(in_)
    layer = Flatten()(layer)
    layer = Dense(256, activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dense(num_classes, activation=None)(layer)
    
    out_ = Activation('softmax')(layer)

    model = Model(in_, out_)

    return model
