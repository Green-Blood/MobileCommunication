import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

Fs = 12
f = 1
sample = 12
x = np.arange(sample)
time = (2 * np.pi * f * x / Fs)

def bpsk_Modulation(bitString):
    signal = np.zeros((len(bitString), 12), float)
    for i in range(len(bitString)):
        if bitString[i] > 0:
            signal[i] = np.sin(time)
        else:
            signal[i] = -np.sin(time)
    return signal

def RGB(photo, channels, height, width):
    bitString = ''
    for i in range(height):
        for j in range(width):
            pixel = photo[i][j]
            for k in range(channels):
                binary = bin(pixel[k])[2:].zfill(8)
                bitString += binary
    return bitString

if __name__ == "__main__":
    # Initializing data
    photo = cv.imread('perduza.jpg')
    height, width, channels = photo.shape
    bitString = RGB(photo, channels, height, width)

    pass