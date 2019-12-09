import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

time = np.arange(0, 2 * np.pi, np.pi / 100)

def RGB(photo, channels, i, j):
    pixel = photo[i][j]
    bitString = ''
    for k in range(channels):
        binary = bin(pixel[k])[2:].zfill(8)
        bitString += binary
    return bitString

def bpsk_Modulation(bitString):
    #Initializing an amplitude
    amplitude = []
    #For the lenght of bit string if it is equal to 1 append positive else negative
    for i in range(len(bitString)):
        if bitString[i] == '1':
            amplitude.append(np.sin(time))
        else:
            amplitude.append(-np.sin(time))
    return amplitude

def avgSigPow(analogS):
    sigpower = 0.0
    for index in range(len(analogS)):
        sigpower = sum([np.power(abs(analogS[index]), 2)])

    return sigpower / len(analogS)
def add_GaussianNoise(analogS, SNRdb):
    noisePower = avgSigPow(analogS) / SNRdb
    noiseSignal = np.sqrt(noisePower) * (np.random.uniform(-1, 1, len(analogS)))
    return noiseSignal

def bpsk_Demodulator(r_bb, L):
    # r_bb = received baseband signal
    x = np.real(r_bb)
    # returns real part of r_bb
    # L oversampling factor
    x = np.convolve(x, np.ones(L))
    # convolve = Discrete linear convolution, np.ones return new array filled with ones
    x = x[np.arange(L, len(x), L)]
    # threshold detector
    ak_cap = (x > 0)
    return ['1' if x else '0' for x in ak_cap]
def bpsk_Demodulation(noisySignal, analogSignal):

    return ''.join(map(str, bpsk_Demodulator(noisySignal + analogSignal, 200)))

def BER_Calculate(bitString, demodulatedBitString):
    # Take a sum of 1 and all not equal bit and dem bit strings, in order to understand the difference
    ber_sum = sum(1 for bitStr, demBitStr in zip(bitString, demodulatedBitString)
        if bitString != demBitStr)
    # add the ber_sum to the difference in lenght of the bit and demdulated bit strings
    ber = ber_sum + abs(len(bitString) - len(demodulatedBitString))

    return ber

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

def plotDemodulatedImage(demodulated_img, name):

    cv2.imwrite(name + ".jpg", demodulated_img)
    img = mpimg.imread(name + '.jpg')
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('After')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

    return


def channel_calculate(SNRdb):
    for i in range(height):
        for j in range(width):
            # Split image to RGB bit string
            bitString = RGB(photo, channels, i, j)
            # Modulate analog signal
            analogSignal = np.array(bpsk_Modulation(bitString)).flatten()
            # Add artificial noise to signal
            noisySignal = add_GaussianNoise(analogSignal, SNRdb)
            # Demodulate the signal
            demodulatedBitString = bpsk_Demodulation(noisySignal, analogSignal)
            demodulated_img[i][j] = [int(demodulatedBitString[0:8], 2), int(demodulatedBitString[8:16], 2),
                                     int(demodulatedBitString[16:24], 2)]
            # Calculate the Bit Error rate
            BER = BER_Calculate(bitString, demodulatedBitString)
    cv2.imwrite("photo_changed_1.jpg", demodulated_img)

    img = mpimg.imread('Peruza.jpg')
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img)
    a.set_title('Before')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

    img = mpimg.imread('photo_changed_1.jpg')
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img)
    imgplot.set_clim(0.0, 0.7)
    a.set_title('After')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

if __name__ == "__main__":
    # Initializing data
    photo = cv2.imread('Peruza.jpg')
    height, width, channels = photo.shape
    BER = 0
    demodulated_img = np.zeros((height, width, channels))

    plotRGBImage()

    # channel_calculate(0)
    channel_calculate(50)
    channel_calculate(100)

    pass
    # print(channels)


