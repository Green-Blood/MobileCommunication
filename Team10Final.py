import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def BER_Calculate(bitString, demodulatedBitString):
    ber_sum = 0
    for i in range(len(bitString)):
        if bitString[i] != demodBitString[i]:
            ber_sum = ber_sum + 1
    # Take a sum of 1 and all not equal bit and dem bit strings, in order to understand the difference
    # ber_sum = sum(1 for bitStr, demBitStr in zip(bitString, demodulatedBitString)
    #               if bitString != demBitStr)
    # add the ber_sum to the difference in lenght of the bit and demdulated bit strings
    ber = ber_sum + abs(len(bitString) - len(demodulatedBitString))

    return ber / len(bitString)


def plotRGBImage():
    # Creating a subplot, to recreate an RGB image
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # In this for, we are going to split them, and show the image
    for c, ax, in zip(range(3), axs):
        # np.zeros Return a new array of given shape and type, filled with zeros.
        tmp_image = np.zeros(photo.shape, dtype="uint8")
        # assign to temporal image R,G and B values(from for c were equal 0,1,2)
        tmp_image[:, :, c] = photo[:, :, c]
        # Show the image
        ax.imshow(tmp_image)
        ax.set_axis_off()


def bpsk_Demodulation(rcvdSignal):
    bitString = np.zeros(len(rcvdSignal), int)

    for i in range(len(rcvdSignal)):

        Zeros = sum([(k ** 2) for k in np.nditer(rcvdSignal[i] - np.sin(time))])
        Ones = sum([(k ** 2) for k in np.nditer(rcvdSignal[i] + np.sin(time))])

        if Ones > Zeros:
            bitString[i] = 1
        else:
            bitString[i] = 0

    return bitString


def add_GaussianNoise(analogS, SNRdb):
    noiseVariance = (10 ** (-SNRdb / 10)) / 2
    noisyS = np.sqrt(noiseVariance) * (np.random.randn(len(analogS), 6))
    rcvdSignal = analogS + noisyS
    return rcvdSignal


def bpsk_Modulation(bitString):
    analogSignal = np.zeros((len(bitString), 6), float)

    for i in range(len(bitString)):
        if bitString[i] > 0:
            analogSignal[i] = np.sin(time)
        else:
            analogSignal[i] = -np.sin(time)

    return analogSignal


def split_to_RGB(photo):
    r = np.array(photo[:, :, 2].flat)
    g = np.array(photo[:, :, 1].flat)
    b = np.array(photo[:, :, 0].flat)
    binString = ''
    rgb = np.concatenate((r, g, b), axis=None)
    for iterator in np.nditer(rgb):
        binString += bin(iterator)[2:].rjust(8, "0")
    bitString = np.array([int(y, 2) for y in binString])
    return bitString


def plotBERgraph(BER, SNRdb_values):
    # x axis values
    x = SNRdb_values
    # corresponding y axis values
    y = BER
    # plotting the points
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('SNRdb')
    # naming the y axis
    plt.ylabel('BER')

    # giving a title to my graph
    plt.title('BER')

    # function to show the plot
    plt.show()


def plotDemodImg(demodBitString, width, heigth):
    redChannel = []
    greenChannel = []
    blueChannel = []

    sizeOfChannel = width * heigth * 8
    bitString = ''.join(str(color) for color in np.nditer(demodBitString))

    for i in range(0, sizeOfChannel, 8):
        redChannel.append(int(bitString[i:i + 8], 2))
        greenChannel.append(int(bitString[(sizeOfChannel + i):(sizeOfChannel + i + 8)], 2))
        blueChannel.append(int(bitString[(2 * sizeOfChannel + i):(2 * sizeOfChannel + i + 8)], 2))
    red = np.array(redChannel, int)
    green = np.array(greenChannel, int)
    blue = np.array(blueChannel, int)

    red = np.reshape(red, (heigth, width))
    green = np.reshape(green, (heigth, width))
    blue = np.reshape(blue, (heigth, width))

    img = np.dstack((blue, green, red))
    return img


if __name__ == "__main__":
    # Initializing data
    photo = cv.imread('Peruza.jpg')
    # Formula for carrier
    time = 2 * np.pi * 1 * np.arange(6) / 6
    height, width, channels = photo.shape
    bitString = split_to_RGB(photo)
    bitString = np.array(list(bitString))
    SNRdb_values = [-10, -5, 0, 5, 10]
    BER = []
    analogSignal = bpsk_Modulation(bitString)
    for SNRdb in SNRdb_values:
        receivedSignal = add_GaussianNoise(analogSignal, SNRdb)
        demodBitString = bpsk_Demodulation(receivedSignal)
        BER.append(BER_Calculate(bitString, demodBitString))
        img = plotDemodImg(demodBitString, width, height)
        img_name = 'DemImg = ' + str(SNRdb) + '.jpg'
        cv.imwrite(img_name, img)
    print(BER)
    plotBERgraph(BER, SNRdb_values)
    plotRGBImage()

pass
