import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import jit, cuda

image = plt.imread("Peruza.jpg")
#Return evenly spaced values within given interval
time = np.arange(0, 2 * np.pi, np.pi / 100)

def bpsk_Modulation(bitString):
    amplitude = []
    #for the whole lenght of bitstring, check if it equal to 0, add to amplitude 0, else 1.
    for i in range(len(bitString)):
        if bitString[i] == '0':
            #Append - sin, to list amplitude
            amplitude.append(-np.sin(time))
        else :
            amplitude.append(np.sin(time))
    return amplitude

def bpsk_Demodulation(r_bb,L):
    # x = real of r_bb
    x = np.real(r_bb)
    # x = convolution of two one-dimensional sequences
    x = np.convolve(x,np.ones(L))
    x = x[np.arange(L,len(x),L)]
    ak_cap = (x > 0)
    return ['1' if x else '0' for x in ak_cap]
#Adding Gaussian Noise
def add_Gaussian_Noise(signal, snr):
    # to get signal power we need to take a sum of
    sigpow = sum([np.power(abs(signal[i]), 2) for i in range(len(signal))])
    print(sigpow)
    sigpow /= len(signal)
    noisepow = sigpow / (np.power(10, snr / 10))
    noise = np.sqrt(noisepow) * (np.random.uniform(-1, 1, size=len(signal)))
    return noise
# Checking bit error rate
def check_BER(bitString, demodBitString):
    return sum(1 for a, b in zip(bitString, demodBitString) if a != b) + abs(len(bitString) - len(demodBitString))
# Function to plot usual image
def plotimage(image, h=8, **kwargs):
    # Helper function to plot an image
    # y and x is positions of a pixel
    y = image.shape[0]
    x = image.shape[1]
    # W is a color value of a pixel
    w = (y / x) * h
    # Below 3 lines are for drawing a picture
    plt.figure(figsize=(w, h))
    plt.imshow(image, interpolation="none", **kwargs)
    plt.axis('off')
def plotRGBImage():
    # Creating a subplot, to recreate an RGB image
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # In this for, we are going to split them, and show the image
    for c, ax, in zip(range(3), axs):
        # np.zeros Return a new array of given shape and type, filled with zeros.
        tmp_image = np.zeros(image.shape, dtype="uint8")
        # assign to temporal image R,G and B values(from for c were equal 0,1,2)
        tmp_image[:, :, c] = image[:, :, c]
        # Show the image
        ax.imshow(tmp_image)
        ax.set_axis_off()


def simulate():
    image = cv2.imread('Peruza.jpg')
    height, width, channels = image.shape
    demod_img = np.zeros((height, width, channels))
    BER = 0
    for i in range(height):
        for j in range(width):
            rgb = image[i][j]
            bitString = ''
            for k in np.arange(3):
                binary = bin(rgb[k])[2:].zfill(8)
                bitString += binary
            signal = np.array(bpsk_Modulation(bitString)).flatten()
            SNR = 0
            nsignal = add_Gaussian_Noise(signal, SNR)
            demodbitString = ''.join(map(str, bpsk_Demodulation(nsignal + signal, 200)));
            demod_img[i][j] = [int(demodbitString[0:8], 2), int(demodbitString[8:16], 2), int(demodbitString[16:24], 2)]
            BER += check_BER(bitString, demodbitString)
    print(BER / (height * width * channels * 8))
    cv2.imwrite("photo_changed_1.jpg", demod_img)

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

    return

simulate()