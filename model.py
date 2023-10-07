import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from time import time
from keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

class SegmentModel():
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        # self.channels = channels
        self.num_classes = num_classes

    def conv_block(self, inputs, n_filters = 32, dropout = 0, max_pooling = True):
        conv = Conv2D(n_filters,
                      kernel_size = 3,
                      activation = 'relu',
                      padding = 'same',
                      kernel_initializer = 'he_normal')(inputs)
        conv = Conv2D(n_filters,
                      kernel_size=3,
                      activation='relu',
                      padding = 'same', 
                      kernel_initializer = 'he_normal')(conv)

        if dropout > 0:
            conv = Dropout(dropout)(conv)

        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2,2))(conv)

        else:
            next_layer = conv

        skip_connection = conv

        return next_layer, skip_connection

    def upsample(self, expansive_input, contractive_input, n_filters = 32):
        pass

    def unet_model(self, n_filters=32, n_classes = 2):
        inputs = Input(self.input_size)

        model = Model(inputs, outputs)
        return model
          

if __name__ == "__main__":
    print(tf.__version__)