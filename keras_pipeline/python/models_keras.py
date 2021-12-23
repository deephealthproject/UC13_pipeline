"""
    Neural Network models definition
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Reshape
from tensorflow.python.keras.layers.core import Dropout, SpatialDropout2D
from tensorflow.python.keras.layers.pooling import MaxPool2D, GlobalMaxPool2D
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam


def create_model(model_id, input_shape, num_classes):
    """
        Base function for building different models.
    """

    if model_id == 'lstm':
        model = build_lstm(input_shape, num_classes)
    elif model_id == 'gru':
        model = build_gru(input_shape, num_classes)
    elif model_id == 'conv':
        model = build_conv(input_shape, num_classes)
    elif model_id == 'conv2':
        model = build_conv2(input_shape, num_classes)
    elif model_id == 'conv-paper':
        model = build_conv_paper(input_shape, num_classes)

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


#-------------------------------------------------------------------------------

def build_conv(input_shape, num_classes):
    
    # input shape : (batch_size, 2560, 23, 1)
    # Input is expected to be a window of 10 seconds

    in_ = Input(shape=input_shape)
    
    k1 = 32
    k2 = 64
    k3 = 128

    layer = GaussianNoise(0.25)(in_)

    layer = Conv2D(k1, kernel_size=(128, 23), strides=(64, 1), padding='valid', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = Reshape(((input_shape[0] - 64) // 64, k1, 1))(layer)
    layer = Conv2D(k2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = SpatialDropout2D(0.25)(layer)
    layer = MaxPool2D()(layer)
    layer = Conv2D(k3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = SpatialDropout2D(0.25)(layer)
    layer = MaxPool2D()(layer)
    layer = Conv2D(k3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = SpatialDropout2D(0.25)(layer)
    layer = MaxPool2D()(layer)
    layer = Conv2D(k3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = SpatialDropout2D(0.25)(layer)

    layer = Flatten()(layer)

    layer = Dense(1024, activation=None)(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(512, activation=None)(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(num_classes, activation=None)(layer)
    out_ = Activation('sigmoid')(layer)

    model = Model(in_, out_)

    return model


# ------------------------------------------------------------------------------


def build_conv2(input_shape, num_classes):
    
    # input shape : (batch_size, 2560, 23, 1)
    # Input is expected to be a window of 10 seconds

    in_ = Input(shape=input_shape)
    
    k = 64

    layer = GaussianNoise(0.25)(in_)

    layer = Conv2D(k, kernel_size=(128, 23), strides=(64, 1), padding='valid', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = Reshape(((input_shape[0] - 64) // 64, k, 1))(layer)

    for k in [128, 128, 256, 256]:
        layer = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_constraint=max_norm(1))(layer)
        layer = Conv2D(k, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
        layer = SpatialDropout2D(0.25)(layer)
        layer = MaxPool2D()(layer)

    layer = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_constraint=max_norm(1))(layer)
    layer = Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_constraint=max_norm(1))(layer)
    layer = SpatialDropout2D(0.25)(layer)
    
    layer = Flatten()(layer)

    layer = Dense(1024, activation=None, kernel_constraint=max_norm(3))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(512, activation=None, kernel_constraint=max_norm(2))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)

    layer = Dense(1, activation=None, kernel_constraint=max_norm(2))(layer)
    out_ = Activation('sigmoid')(layer)

    model = Model(in_, out_)

    return model


# ------------------------------------------------------------------------------

def conv_block(layer_in, filters):

    layer = Conv2D(filters, kernel_size=(3, 1))(layer_in)
    layer = Conv2D(filters, kernel_size=(1, 2))(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    return layer


def build_conv_paper(input_shape, num_classes):
    
    # input shape : (batch_size, 256, 23, 1)
    # Input is expected to be a window of 1 second

    in_ = Input(shape=input_shape)

    layer = conv_block(in_, 16)
    layer = MaxPool2D((2, 1))(layer)

    layer = conv_block(layer, 16)
    layer = MaxPool2D((2, 1))(layer)

    layer = conv_block(layer, 32)
    layer = MaxPool2D((2, 1))(layer)

    layer = conv_block(layer, 32)
    layer = MaxPool2D((2, 2))(layer)

    layer = conv_block(layer, 64)
    layer = MaxPool2D((2, 2))(layer)

    layer = conv_block(layer, 64)
    #layer = MaxPool2D((2, 2))(layer)

    #layer = conv_block(layer, 16)
    #layer = MaxPool2D((2, 2))(layer)

    layer = Flatten()(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(num_classes, activation=None)(layer)
    
    out_ = Activation('sigmoid')(layer)

    model = Model(in_, out_)

    return model



if __name__=='__main__':
    net = build_conv2(input_shape=(256, 23, 1), num_classes=2)
    net.compile(optimizer=Adam(),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                    )
    net.summary()
