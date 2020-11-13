# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte
import sys
import time
import struct
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import *

def main():
    if(len(sys.argv) != 3):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    if(sys.argv[1] != "-d"):
        sys.exit("Wrong or missing parameter. Please execute with: -d dataset")
    dataset = sys.argv[2]

    with open(dataset, "rb") as file:
        magic_num = int.from_bytes(file.read(4), byteorder='big')
        images = int.from_bytes(file.read(4), byteorder='big')
        rows = int.from_bytes(file.read(4), byteorder='big')
        columns = int.from_bytes(file.read(4), byteorder='big')
        print("Begin autoencoder with:    (magic_num)->"+str(magic_num)+"    (images)->"+str(images)+"    (rows)->"+str(rows)+"    (cols)->"+str(columns))
        print("Store data in array...")
        # 2d numpy array for images->pixels
        pixels = np.array(list(bytes_group(rows*columns, file.read(), fillvalue=0)))
        print("Data ready in numpy array! Begin building model...")

        input_img = Input(shape=(28, 28, 1))
        # input_img = np.resize(pixels,(28,28,1))
        autoencoder = Model(input_img, decoder(encoder(input_img)))
        autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
        train_X,valid_X,train_ground,valid_ground = train_test_split(pixels, pixels, test_size=0.2, random_state=13)
        # autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=128,epochs=5,verbose=1,validation_data=(valid_X, valid_ground))

        # User choices:
        # run_again = 1
        # while(run_again>0 and run_again<4):
        #     try:
        #         run_again = int(input("\nChoose one from below options(1-4): \n1)Execute program with different parameters\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n\n"))
        #     except:
        #         print("Invalid choice.Try again\n")
start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
