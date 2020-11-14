# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte
import sys
import time
import struct
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from functions import *

# A part hyperparameters -> number of layers, filter size, number of filters/layer, number of epochs, batch size
def main():
    if(len(sys.argv) != 3):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    if(sys.argv[1] != "-d"):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    dataset = sys.argv[2]

    # numarray[0] -> magic_number, [1] -> images, [2] -> rows, [3] -> columns
    pixels, numarray = numpy_from_dataset(dataset, 4)
    if(len(numarray)!=4 or len(pixels)==0):
        sys.exit("Input dataset does not have the required number of values")
    train_X, valid_X, train_Y, valid_Y = reshape_dataset(pixels, numarray)
    print("Data ready in numpy array!")

    while(True):
        print("Begin building model...")
        input_img = Input(shape=(numarray[2], numarray[3], 1))
        autoencoder = Model(input_img, decoder(encoder(input_img, 4, 32), 4, 32))
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        autoencoder_train = autoencoder.fit(train_X, train_Y, batch_size=124, epochs=5, verbose=1, validation_data=(valid_X, valid_Y))
        # train_ER = autoencoder.evaluate(train_X, train_Y, verbose=1)

        # autoencoder = load_model("autoencoder")
        # autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

        # User choices:
        parameters, continue_flag = user_choices(autoencoder, "autoencoder", autoencoder_train)
        if(not continue_flag):
            break;

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
