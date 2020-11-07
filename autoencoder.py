# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte
import sys
import time
import struct
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
        pixels = list(bytes_group(rows*columns, file.read(), fillvalue=0))

        print(magic_num, images, rows, columns)
        # print(pixels[-1])

        # User choices:
        # run_again = 1
        # while(run_again>0 and run_again<4):
        #     try:
        #         run_again = int(input("Choose one from below options(1-4): \n1)Execute program with different parameters\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n\n"))
        #     except:
        #         print("Invalid choice.Try again\n")


start_time = time.time()
main()
print("\nExecution time: %s seconds" % (time.time() - start_time))
