import sys
import time
import struct
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from functions import *

# A part hyperparameters -> number of layers, filter size, number of filters/layer, number of epochs, batch size
#python3 classification.py -d ./Datasets/train-images-idx3-ubyte -dl ./Datasets/train-labels-idx1-ubyte -t ./Datasets/t10k-images-idx3-ubyte -tl ./Datasets/t10k-labels-idx1-ubyte -model autoencoder

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
            trainlabels = sys.argv[i+1]
        elif sys.argv[i] == "-t":
            argcheck[2] = True
            testset = sys.argv[i + 1]
        elif sys.argv[i] == "-tl":
            argcheck[3] = True
            testlabels = sys.argv[i + 1]
        elif sys.argv[i] == "-model":
            argcheck[4] = True
            autoencoder = sys.argv[i + 1]
    for i in range(0, 5):
        if argcheck[i] is False:
            sys.exit("Wrong or missing parameter. Please execute with –d <training set> –dl <traininglabels> -t <testset> -tl <test labels> -model <autoencoder h5>")

    train_pixels, train_numarray = numpy_from_dataset(trainset, 4)
    train_pixels = np.reshape(train_pixels.astype('float32') / 255., (-1, train_numarray[2], train_numarray[3]))

    train_labels, train_labels_numarray = numpy_from_dataset(trainlabels, 2)
    print(len(train_labels),train_labels[0])
    test_pixels, test_numarray = numpy_from_dataset(testset, 4)
    test_pixels = np.reshape(test_pixels.astype('float32') / 255., (-1, test_numarray[2], test_numarray[3]))

    test_labels, test_labels_numarray = numpy_from_dataset(testlabels, 2)

    # binary_train_label = []
    # for i, label in enumerate(train_labels, start=0):
    #     temp = []
    #     for j in range(0,label[0]):
    #         temp.append(0)
    #     temp.append(1)
    #     for z in range(0,10 - label[0] -1):
    #         temp.append(0)
    #     binary_train_label.append(temp)
    # binary_train_label = np.array(binary_train_label)
    #
    # binary_test_label = []
    # for i, label in enumerate(test_labels, start=0):
    #     temp = []
    #     for j in range(0, label[0]):
    #         temp.append(0)
    #     temp.append(1)
    #     for z in range(0, 10 - label[0] -1):
    #         temp.append(0)
    #     binary_test_label.append(temp)
    # binary_test_label = np.array(binary_test_label)
    binary_train_label = labels_to_binary(train_labels, 10)
    binary_test_label = labels_to_binary(test_labels, 10)

    print(binary_train_label[0], binary_train_label.shape, train_labels[0])
    # test_labels = np.reshape(test_labels.astype('float32') / 255., (-1, test_labels_numarray[1], 10))
    # train_labels = np.reshape(train_labels.astype('float32') / 255., (-1,train_labels_numarray[1], 10))

    if len(train_numarray) != 4 or len(train_pixels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(train_labels_numarray) != 2 or len(train_labels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(test_numarray) != 4 or len(test_pixels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(test_labels_numarray) != 2 or len(test_labels) == 0:
        sys.exit("Input dataset does not have the required number of values")

    # train_X, valid_X, train_Y, valid_Y = reshape_dataset(pixels, numarray)

    # print("train_X ", len(train_X[0]), len(train_X))
    # print("numarray ", numarray[0], len(numarray))

    #convert label set to boolean labels

    print("Data ready in numpy array!\n")

    parameters = [4, 3, 32, 200, 80]
    newparameter = [[] for i in range(len(parameters))]

    while True:
        input_img = Input(shape=(train_numarray[2], train_numarray[3], 1))

        # encoding = Model(input_img, encoder(input_img, parameters))
        # encoding.summary()
        # print(encoding.layers)

        # load autoencoder
        autoencoderModel = load_model(autoencoder)
        autoencoderModel.load_weights(autoencoder + ".h5")
        autoencoderModel.compile(loss='mean_squared_error', optimizer=RMSprop())
        autoencoderModel.summary()

        # create classifier Model
        classifier = Model(inputs=input_img,
                           outputs=classifier_layers(autoencoderModel, count_half_layers(parameters[0]), input_img))

        # for layer in classifier.layers[1:count_half_layers(parameters[0])]:
        #     layer.trainable = False

        classifier.compile(loss='mean_squared_error', optimizer=RMSprop())
        classifier.summary()

        # classifier training
        train_time = time.time()
        classifier_train = classifier.fit(train_pixels, binary_train_label, batch_size=parameters[4], epochs=parameters[3], verbose=1, validation_data=(test_pixels, binary_test_label))
        train_time = time.time() - train_time
        classifier.evaluate(test_pixels, binary_test_label, verbose=1)

        # User choices:
        parameters, continue_flag = user_choices(classifier, classifier_train, parameters, train_time, newparameter)
        if not continue_flag:
            break

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
