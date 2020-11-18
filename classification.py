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
    argcheck = []
    for i in range(0, 5):
        argcheck.append(False)
    print(sys.argv)
    if len(sys.argv) != 11:
        sys.exit("Wrong or missing parameter. Please execute with –d <training set> –dl <traininglabels> -t <testset> -tl <test labels> -model <autoencoder h5>")
    for i in range(0, 11):
        if sys.argv[i] == "-d":
            argcheck[0] = True
            trainset = sys.argv[i+1]
        elif sys.argv[i] == "-dl":
            argcheck[1] = True
            train_labels = sys.argv[i+1]
        elif sys.argv[i] == "-t":
            argcheck[2] = True
            testset = sys.argv[i + 1]
        elif sys.argv[i] == "-tl":
            argcheck[3] = True
            test_labels = sys.argv[i + 1]
        elif sys.argv[i] == "-model":
            argcheck[4] = True
            autoencoder = sys.argv[i + 1]
    for i in range(0, 5):
        if argcheck[i] is False:
            sys.exit("Wrong or missing parameter. Please execute with –d <training set> –dl <traininglabels> -t <testset> -tl <test labels> -model <autoencoder h5>")

    # numarray[0] -> magic_number, [1] -> images, [2] -> rows, [3] -> columns
    # pixels, numarray = numpy_from_dataset(dataset, 4)
    # if(len(numarray)!=4 or len(pixels)==0):
    #     sys.exit("Input dataset does not have the required number of values")
    # train_X, valid_X, train_Y, valid_Y = reshape_dataset(pixels, numarray)
    # print("Data ready in numpy array!")

    pixels, numarray = numpy_from_dataset(trainset, 4)
    if len(numarray) != 4 or len(pixels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    parameters = [4, 3, 8, 5, 100]
    input_img = Input(shape=(numarray[2], numarray[3], 1))
    encoding = Model(input_img, encoder(input_img, parameters))
    encoding.summary()
    # print(encoding.layers)

    encoderModel = load_model(autoencoder)
    encoderModel.load_weights(autoencoder + ".h5")
    encoderModel.compile(loss='mean_squared_error', optimizer=RMSprop())
    encoderModel.summary()
    for layer in encoderModel.layers[10:17]:
        layer.trainable = False
    encoderModel.summary()

    x = encoderModel(input_img)
    output = fcTraining(x)

    model = Model(inputs=input_img, outputs=output)
    print(model.summary())
    #plot_model(model, to_file='convolutional_neural_network.png')


    # while(True):
        # print("Begin building model...")
        # input_img = Input(shape=(numarray[2], numarray[3], 1))
        # autoencoder = Model(input_img, decoder(encoder(input_img, 4, 32), 4, 32))
        # autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        # autoencoder_train = autoencoder.fit(train_X, train_Y, batch_size=32, epochs=5,verbose=1,validation_data=(valid_X, valid_Y))

        # autoencoder = load_model("autoencoder")
        # autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

        # User choices:
        # parameters, continue_flag = user_choices(autoencoder, "autoencoder")
        # if(not continue_flag):
        #     break;

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
