import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

time = np.arange(0, 2 * np.pi, np.pi / 100)


def bpsk_Modulation(bitString):
    amplitude = []
    for i in range(len(bitString)):
        if bitString[i] == '1':
            amplitude.append(np.sin(time))
        else:
            amplitude.append(-np.sin(time))
    return amplitude


def plotImaging(analogS):
    return


def bpsk_Demodulation(r_bb, L):
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

def BER(binString, demodBinString):
    return sum(1 for a, b in zip(binString, demodBinString) if a != b) + abs(len(binString) - len(demodBinString))


def avgSigPow(analogS):
    sigpower = 0.0
    for index in range(len(analogS)):
        sigpower = sum([np.power(abs(analogS[index]), 2)])

    return sigpower / len(analogS)


def add_GaussianNoise(analogS, SNRdb):
    noisePower = avgSigPow(analogS) / SNRdb
    # sigpower = sum([np.power(abs(analogS[i]),2) for i in range(len(analogS))])
    # sigpower = sigpower/len(analogS)
    # noisepower=sigpower/(np.power(10,SNRdb/10))
    noiseSignal = np.sqrt(noisePower) * (np.random.uniform(-1, 1, len(analogS)))
    return noiseSignal


def RGB(photo, channels, i, j):
    pixel = photo[i][j]
    bitString = ''
    for k in range(channels):
        binary = bin(pixel[k])[2:].zfill(8)
        bitString += binary
    return bitString


if __name__ == "__main__":

    SNRdb = 10
    photo = cv2.imread('Peruza.jpg')
    height, width, channels = photo.shape
    # print(channels)
    for i in range(height):
        for j in range(width):
            bitString = RGB(photo, channels, i, j)
            analogSignal = np.array(bpsk_Modulation(bitString)).flatten()
            noisySignal = add_GaussianNoise(analogSignal, SNRdb)
            demodulatedBitString = ''.join(map(str, bpsk_Demodulation(noisySignal + analogSignal, 200)))
            print("ARRB")

    pass