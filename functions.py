import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

def numpy_from_dataset(inputpath, numbers):
    pixels = []
    numarray = []
    with open(inputpath, "rb") as file:
        for x in range(numbers):
            numarray.append(int.from_bytes(file.read(4), byteorder='big'))
        # print("Begin autoencoder with: "+str())
        print("Storing data in array...")
        # 2d numpy array for images->pixels
        if numbers == 4:
            pixels = np.array(list(bytes_group(numarray[2]*numarray[3], file.read(), fillvalue=0)))
        elif numbers == 2:
            # pixels = np.array(list(bytes_group(rows*columns, file.read(), fillvalue=0)))
            print(numarray[0], " ", numarray[1])
    return pixels, numarray

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

def encoder(input_img, layers, filters):
    mask_size = 3
    conv = input_img
    # for i in range(layers):
    #     conv = Conv2D(filters, (mask_size, mask_size), activation='relu', padding='same')(conv)
    #     conv = BatchNormalization()(conv)
    #     conv = MaxPooling2D(pool_size=(2,2))(conv)
    #     print("Encoder: ", i, filters)
    #     filters*=2
    # return conv

    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 8
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                           #14 x 14 x 8
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)    #14 x 14 x 16
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                           #7 x 7 x 16
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)    #7 x 7 x 32
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)    #7 x 7 x 64
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv, layers, filters):
    mask_size = 3
    filters = filters*pow(2,layers-1)
    # for i in range(layers):
    #     conv = Conv2D(filters, (mask_size, mask_size), activation='relu', padding='same')(conv)
    #     conv = BatchNormalization()(conv)
    #     conv = UpSampling2D((2,2))(conv)
    #     print("Decoder: ", i, filters)
    #     filters/=2
    # conv = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv)
    # return conv

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv)     #7 x 7 x 64
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)    #7 x 7 x 32
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv5)    #7 x 7 x 16
    conv7 = BatchNormalization()(conv7)
    up1 = UpSampling2D((2,2))(conv7)                                        #14 x 14 x 16
    conv8 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)       #14 x 14 x 8
    conv8 = BatchNormalization()(conv8)
    up2 = UpSampling2D((2,2))(conv8)                                        #28 x 28 x 8
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  #28 x 28 x 1
    return decoded

def save_model(model, modelname):
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

def error_graphs(modeltrain):
    plt.plot(modeltrain.history['loss'], label='train')
    plt.plot(modeltrain.history['val_loss'], label='test')
    plt.title('Loss / Mean Squared Error')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()

def reshape_dataset(dataset, numarray):
    train_X, valid_X, train_Y, valid_Y = train_test_split(dataset, dataset, test_size=0.2, random_state=13)
    # Reshapes to (x, rows, columns)
    train_X = np.reshape(train_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_X = np.reshape(valid_X.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    train_Y = np.reshape(train_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    valid_Y = np.reshape(valid_Y.astype('float32') / 255., (-1, numarray[2], numarray[3]))
    return train_X, valid_X, train_Y, valid_Y

def user_choices(model, modelname, modeltrain):
    parameters = []
    continue_flag = True
    while(True):
        try:
            parameters.clear()
            run_again = int(input("\nUSER CHOICES: choose one from below options(1-4): \n1)Execute program with different hyperparameters\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n---------------> "))
        except:
            print("Invalid choice.Try again\n")

        if(run_again==1):
            parameters.append(int(input("Type number of layers: ")))
            parameters.append(int(input("Type filter size: ")))
            parameters.append(int(input("Type number of filters/layer: ")))
            parameters.append(int(input("Type number of epochs: ")))
            parameters.append(int(input("Type batch size: ")))
            break;
        elif(run_again == 2):
            error_graphs(modeltrain)
        elif(run_again == 3):
            save_model(model, modelname)
        elif(run_again == 4):
            continue_flag = False
            print("Program terminates...\n")
            break;
        else:
            print("Invalid choice.Try again\n")

    return parameters, continue_flag;
