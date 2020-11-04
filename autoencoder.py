# python autoencoder.py -d ./Datasets/train-images-idx3-ubyte
import sys
import struct

def main():
    if(len(sys.argv) == 3):
        if(sys.argv[1] == "-d"):
            dataset = sys.argv[2]
            with open(dataset, "rb") as file:
                magic_num = int.from_bytes(file.read(4), byteorder='big')
                images = int.from_bytes(file.read(4), byteorder='big')
                rows = int.from_bytes(file.read(4), byteorder='big')
                columns = int.from_bytes(file.read(4), byteorder='big')

                pixels = []
                while (bytes := file.read(rows*columns)):
                    pixels.append(bytes)
                print(magic_num, images, rows, columns, len(pixels))

                # User choices:
                # run_again = 1
                # while(run_again>0 and run_again<4):
                #     try:
                #         run_again = int(input("Choose one from below options(1-4): \n1)Execute program with different parameters\n2)Show error-graphs\n3)Save the existing model\n4)Exit\n\n"))
                #     except:
                #         print("Invalid choice.Try again\n")
        else:
            print("Missing dataset parameter. Please execute with: -d dataset")
    else:
        print("Missing dataset parameter. Please execute with: -d dataset")

main()
