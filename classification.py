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
    test_pixels, test_numarray = numpy_from_dataset(testset, 4)
    test_pixels = np.reshape(test_pixels.astype('float32') / 255., (-1, test_numarray[2], test_numarray[3]))

    test_labels, test_labels_numarray = numpy_from_dataset(testlabels, 2)

    # fix labels from lists to ints
    temp = []
    for label in train_labels:
        temp.append(label[0])
    train_labels = np.array(temp)

    temp = []
    for label in test_labels:
        temp.append(label[0])
    test_labels = np.array(temp)

    binary_train_label = labels_to_binary(train_labels, 10)
    binary_test_label = labels_to_binary(test_labels, 10)

    train_X, valid_X, train_label, valid_label = train_test_split(train_pixels, binary_train_label, test_size=0.2, random_state=13)

    print("Original label: ", train_labels[0])
    print('After conversion to one-hot: ', binary_train_label[0])

    if len(train_numarray) != 4 or len(train_pixels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(train_labels_numarray) != 2 or len(train_labels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(test_numarray) != 4 or len(test_pixels) == 0:
        sys.exit("Input dataset does not have the required number of values")
    if len(test_labels_numarray) != 2 or len(test_labels) == 0:
        sys.exit("Input dataset does not have the required number of values")

    #convert label set to boolean labels

    print("Data ready in numpy array!\n")
    df = classification_values_df()
    hypernames = ["Layers", "Fc_units", "Epochs", "Batch_Size"]
    #parameters = [4, 32, 20, 64]
    parameters = classification_input_parameters()
    newparameter = [[] for i in range(len(parameters))]
    originparms = parameters.copy()
    oldparm = -1
    layers_check = parameters[0]
    while True:
        input_img = Input(shape=(train_numarray[2], train_numarray[3], 1))
        if parameters[0] != layers_check:
            layers_check = parameters[0]
            autoencoder = input("Type new autoencoder with the same layers: ")
        # load autoencoder
        autoencoderModel = load_model(autoencoder)
        autoencoderModel.load_weights(autoencoder + ".h5")
        autoencoderModel.compile(loss='mean_squared_error', optimizer=RMSprop())
        autoencoderModel.summary()

        # create classifier Model
        classifier = Model(inputs=input_img,
                           outputs=classifier_layers(autoencoderModel, count_half_layers(parameters[0]), parameters[1], input_img))

        for layer in classifier.layers[1:count_half_layers(parameters[0])]:
            layer.trainable = False

        classifier.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
        classifier.summary()

        # classifier training
        train_time = time.time()
        classifier_train = classifier.fit(train_X, train_label, batch_size=parameters[3], epochs=parameters[2], verbose=1, validation_data=(valid_X, valid_label))
        classifier.save_weights('first_part_classification.h5')

        for layer in classifier.layers[1:count_half_layers(parameters[0])]:
            layer.trainable = True

        classifier.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])

        classifier_train = classifier.fit(train_X, train_label, batch_size=parameters[3], epochs=parameters[2],
                                          verbose=1, validation_data=(valid_X, valid_label))
        classifier.save_weights('classification.h5')

        #loss and accuracy plot
        training_plots(classifier)

        eval = classifier.evaluate(test_pixels, binary_test_label, verbose=0)
        print('Test loss: ', eval[0], 'Test accuracy: ', eval[1])

        #prediction
        predicted = classifier.predict(test_pixels)

        predicted_labels = np.round(predicted)
        temp = []
        for array in predicted_labels:
            temp.append(np.argmax(array))
        predicted_labels = np.array(temp)

        print_predictions_numbers(test_labels, predicted_labels)
        print(classification_report(test_labels, predicted_labels))

        train_time = time.time() - train_time

        # User choices:
        parameters, continue_flag, oldparm = user_choices_classification(classifier, classifier_train, parameters, originparms, train_time, newparameter, oldparm,df,hypernames, test_pixels, test_labels, predicted_labels)
        if not continue_flag:
            break

start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
