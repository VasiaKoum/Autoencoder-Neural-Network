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
            pixels = np.array(list(bytes_group(1, file.read(), fillvalue=0)))
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

def error_graphs(modeltrain, parameters, train_time, newparameter, indexparm, originparms):
    loss = []
    val = []
    values = []
    times = []
    for i in range(len(newparameter)):
        loss.clear()
        val.clear()
        times.clear()
        values.clear()
        for j in newparameter[i]:
            values.append(j[0])
            loss.append(j[1])
            val.append(j[2])
            times.append(j[3])
        if (i == indexparm-1):
            values.append(parameters[indexparm-1])
            loss.append(modeltrain.history['loss'][-1])
            val.append(modeltrain.history['val_loss'][-1])
        if newparameter[i]:
            graphname = name_parameter(originparms, i, True) + ".png"
            plt.plot(values, loss, label='train', linestyle='dashed', linewidth = 3,  marker='o', markersize=9)
            plt.plot(values, val, label='test', linestyle='dashed', linewidth = 3,  marker='o', markersize=9)
            plt.title('Loss / Mean Squared Error in '+str(round(times[-1], 3))+'sec')
            plt.ylabel('Loss')
            plt.xlabel(name_parameter(originparms, i, False))
            plt.legend(['loss', 'val_loss'], loc='upper left')
            print("Save graph with name: ",graphname)
            plt.savefig(graphname)
            plt.show()
            plt.close()
    return

def name_parameter(parameters, number, flag):
    name = ""
    if (flag):
        if (number==0):
            name = "Lx"+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])
        elif (number==1):
            name = "L"+str(parameters[0])+"_FSx"+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_B"+str(parameters[4])
        elif (number==2):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FLx"+"_E"+str(parameters[3])+"_B"+str(parameters[4])
        elif (number==3):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_Ex"+"_B"+str(parameters[4])
        elif (number==4):
            name = "L"+str(parameters[0])+"_FS"+str(parameters[1])+"_FL"+str(parameters[2])+"_E"+str(parameters[3])+"_Bx"
    else:
        if (number==0):
            name = "Layers"
        elif (number==1):
            name = "Filter_Size"
        elif (number==2):
            name = "Filters/Layer"
        elif (number==3):
            name = "Epochs"
        elif (number==4):
            name = "Batch_Size"
    return name

def reshape_dataset(dataset, numarray):
    train_X, valid_X, train_Y, valid_Y = train_test_split(dataset, dataset, test_size=0.2, random_state=13)
    # Reshapes to (x, rows, columns)
    train_X = np.reshape(train_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_X = np.reshape(valid_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    train_Y = np.reshape(train_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_Y = np.reshape(valid_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    return train_X, valid_X, train_Y, valid_Y

# def parameters_update(oldparm, indexparm, parameters, originparms, modeltrain, newparameter, changepar):
#     tmpparm = oldparm
#     if tmpparm<0:
#         tmpparm = indexparm
#     tmp = [parameters[tmpparm-1]] + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]]
#     newparameter[tmpparm-1].append(tmp)
#     parameters = originparms.copy()
#     parameters[indexparm-1] = changepar
#     oldparm = indexparm
#     return newparameter, parameters, oldparm

def user_choices(model, modeltrain, parameters, originparms, train_time, newparameter, oldparm):
    continue_flag = True
    while (True):
        try:
            run_again = int(input("\nUSER CHOICES: choose one from below options(1-4): \n1)Execute program with different hyperparameter\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n---------------> "))
        except:
            print("Invalid choice.Try again\n")
        if (run_again==1):
            try:
                indexparm = int(input("Choose what parameter would like to change (options 1-5): \n1)Layers\n2)Filter size\n3)Filters/Layer\n4)Epochs\n5)Batch size\n---------------> "))
            except:
                print("Invalid choice.Try again\n")
            # FIX THIS
            if (indexparm>=1 and indexparm<=5):
                try:
                    changepar = int(input("Number for "+ name_parameter(parameters, indexparm-1, False) +" is "+str(parameters[indexparm-1])+". Type the new number: "))
                except:
                    print("Invalid choice.Try again\n")
                    break
                tmpparm = oldparm
                if tmpparm<0:
                    tmpparm = indexparm
                tmp = [parameters[tmpparm-1]] + [modeltrain.history['loss'][-1]] + [modeltrain.history['val_loss'][-1]] + [train_time]
                newparameter[tmpparm-1].append(tmp)
                parameters = originparms.copy()
                parameters[indexparm-1] = changepar
                oldparm = indexparm
                # newparameter, parameters, oldparm = parameters_update(oldparm, indexparm, parameters, originparms, modeltrain, newparameter, changepar)
                break;
            else:
                print("Invalid choice.Try again\n")
        elif (run_again == 2):
            error_graphs(modeltrain, parameters, train_time, newparameter, oldparm, originparms)
            continue_flag = False
            break
        elif (run_again == 3):
            save_model(model)
            continue_flag = False
            break
        elif (run_again == 4):
            continue_flag = False
            print("Program terminates...\n")
            break
        else:
            print("Invalid choice.Try again\n")
    return parameters, continue_flag, oldparm;

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



def encoder_layers(autoencoder, autoencoderLayers, input):
    encoderLayers = autoencoderLayers
    x = input
    for layer in autoencoder.layers[1:encoderLayers]:
        x = layer(x)
    return x


def fc_layers(input):
    x = Flatten()(input)
    x = Dense(32, activation='relu')(x)
    #x = Dropout(0.2,input_shape=(28,))(x)
    x = Dense(10, activation='softmax')(x)
    print("train fully connected layers")
    return x


def classifier_layers(autoencoder, autoencoderLayers, input):
    # encoder_layers
    x = encoder_layers(autoencoder, autoencoderLayers, input)
    # fully connected layers
    x = fc_layers(x)
    return x


def count_half_layers(layers):
    result = layers*3 - 1
    print("layers of encoder:", result)
    return result


def labels_to_binary(labels, categ_range):
    binary_labels = []
    for i, label in enumerate(labels, start=0):
        temp = []
        for j in range(0, label[0]):
            temp.append(0)
        temp.append(1)
        for z in range(0, categ_range - label[0] - 1):
            temp.append(0)
        binary_labels.append(temp)
    binary_labels = np.array(binary_labels)
    return binary_labels
