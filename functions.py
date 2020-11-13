from itertools import zip_longest
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)

def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small & thick)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small & thick)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(convenc):
    #decoder
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(convenc) #7 x 7 x 256
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 128
    conv6 = BatchNormalization()(conv6)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv7 = BatchNormalization()(conv7)
    up1 = UpSampling2D((2,2))(conv7) #14 x 14 x 64
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv8 = BatchNormalization()(conv8)
    up2 = UpSampling2D((2,2))(conv8) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded
