import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Flatten, Dense

def numpy_from_dataset(inputpath, numbers):
    pixels = []
    numarray = []
    with open(inputpath, "rb") as file:
        for x in range(numbers):
            numarray.append(int.from_bytes(file.read(4), byteorder='big'))
        print("Storing data in array...")
        # 2d numpy array for images->pixels
        if numbers == 4:
            pixels = np.array(list(bytes_group(numarray[2]*numarray[3], file.read(), fillvalue=0)))
        elif numbers == 2:
            # FIX THIS
            # pixels = np.array(list(bytes_group(rows*columns, file.read(), fillvalue=0)))
            print(numarray[0], " ", numarray[1])
    return pixels, numarray

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

def encoder(input_img, parameters):
    layers = parameters[0]
    filter_size = parameters[1]
    filters = parameters[2]
    conv = input_img
    for i in range(layers):
        conv = Conv2D(filters, (filter_size, filter_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if (i<2):
            conv = MaxPooling2D(pool_size=(2,2))(conv)
        conv = Dropout(0.2)(conv)
        filters*=2
    return conv

def decoder(conv, parameters):
    layers = parameters[0]
    filter_size = parameters[1]
    filters = parameters[2]*pow(2,parameters[0]-1)
    for i in range(layers):
        conv = Conv2D(filters, (filter_size, filter_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if (i>=layers-2):
            conv = UpSampling2D((2,2))(conv)
        filters/=2
    conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv)
    return conv

def save_model(model, modelname):
    modelname = input("Type the name for model(without extension eg.h5): ")
    print("Saving Model: "+modelname+".json & "+modelname+".h5...")
    # Save model in JSON file
    model_json = model.to_json()
    with open(modelname+".json", "w") as json_file:
        json_file.write(model_json)
    # Save weights from model in h5 file
    model.save_weights(modelname+".h5")

def load_model(modelname):
    # Load model from JSON file
    print("Loading Model: "+modelname+".json & "+modelname+".h5...")
    json_file = open(modelname+".json", 'r')
    autoencoder_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(autoencoder_json)
    # Load weights from h5 file
    autoencoder.load_weights(modelname+".h5")
    return autoencoder

def error_graphs(modeltrain, parameters, train_time):
    print(modeltrain.history['loss'][-1])
    print(modeltrain.history['val_loss'][-1])
    plt.plot(modeltrain.history['loss'], label='train')
    plt.plot(modeltrain.history['val_loss'], label='test')
    plt.title('Loss / Mean Squared Error in '+str(round(train_time, 3))+'sec')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    graphname = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])+".png"
    print("Save graph with name: ",graphname)
    plt.savefig(graphname)
    plt.show()
    plt.close()

def reshape_dataset(dataset, numarray):
    train_X, valid_X, train_Y, valid_Y = train_test_split(dataset, dataset, test_size=0.2, random_state=13)
    # Reshapes to (x, rows, columns)
    train_X = np.reshape(train_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_X = np.reshape(valid_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    train_Y = np.reshape(train_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_Y = np.reshape(valid_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    return train_X, valid_X, train_Y, valid_Y

def user_choices(model, modeltrain, parameters, train_time, newparameter):
    continue_flag = True
    while(True):
        try:
            run_again = int(input("\nUSER CHOICES: choose one from below options(1-4): \n1)Execute program with different hyperparameter\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n---------------> "))
        except:
            print("Invalid choice.Try again\n")
        if(run_again==1):
            try:
                indexpar = int(input("Choose what parameter would like to change (options 1-5): \n1)Layers\n2)Filter size\n3)Filters/Layer\n4)Epochs\n5)Batch size\n---------------> "))
            except:
                print("Invalid choice.Try again\n")
            # FIX THIS
            if(indexpar>=1 and indexpar<=5):
                message = "Number for this parameter is "+str(parameters[indexpar-1])+". Type the new number: "
                changepar = int(input("Number for this parameter is "+str(parameters[indexpar-1])+". Type the new number: "))
                tmp = parameters + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]]
                newparameter[indexpar-1].append(tmp)
                print(np.matrix(newparameter))
                parameters[indexpar-1] = changepar
                break;
            else:
                print("Invalid choice.Try again\n")
        elif(run_again == 2):
            error_graphs(modeltrain, parameters, train_time)
        elif(run_again == 3):
            save_model(model)
        elif(run_again == 4):
            continue_flag = False
            print("Program terminates...\n")
            break;
        else:
            print("Invalid choice.Try again\n")
    return parameters, continue_flag;

def input_parameters():
    parameters = []
    try:
        parameters.append(int(input("Type number of layers: ")))
        parameters.append(int(input("Type filter size: ")))
        parameters.append(int(input("Type number of filters/layer: ")))
        parameters.append(int(input("Type number of epochs: ")))
        parameters.append(int(input("Type batch size: ")))
    except:
        print("Invalid choice.Try again\n")
    return parameters

def fcTraining(model):
    print("train fully connected layers")
