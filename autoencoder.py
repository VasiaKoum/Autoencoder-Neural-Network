# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte
import sys
import time
import struct
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from functions import *

# Autoencoder hyperparameters -> number of layers, filter size, number of filters/layer, number of epochs, batch size
def main():
    if (len(sys.argv) != 3):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    if (sys.argv[1] != "-d"):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")

    dataset = sys.argv[2]
    # numarray[0] -> magic_number, [1] -> images, [2] -> rows, [3] -> columns
    df = values_df()
    hypernames = ["Layers", "Filter_Size", "Filters/Layer", "Epochs", "Batch_Size"]
    pixels, numarray = numpy_from_dataset(dataset, 4)
    if (len(numarray)!=4 or len(pixels)==0):
        sys.exit("Input dataset does not have the required number of values")
    train_X, valid_X, train_Y, valid_Y = reshape_dataset(pixels, numarray)
    print("Data ready in numpy array!\n")
    # Layers, Filter_size, Filters/Layer, Epochs, Batch_size
    parameters = input_parameters()
    newparameter = [[] for i in range(len(parameters))]
    originparms = parameters.copy()
    oldparm = -1

    while (True):
        print("\nBegin building model...")
        input_img = Input(shape=(numarray[2], numarray[3], 1))
        autoencoder = Model(input_img, decoder(encoder(input_img, parameters), parameters))
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        train_time = time.time()
        autoencoder_train = autoencoder.fit(train_X, train_Y, batch_size=parameters[4], epochs=parameters[3], verbose=1, validation_data=(valid_X, valid_Y))
        train_time = time.time() - train_time
        print(autoencoder.summary())

        # User choices:
        parameters, continue_flag, oldparm = user_choices(autoencoder, autoencoder_train, parameters, originparms, train_time, newparameter, oldparm, df, hypernames)
        if (not continue_flag):
            break;

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
