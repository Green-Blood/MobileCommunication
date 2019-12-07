import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import jit, cuda


#Return evenly spaced values within given interval
time = np.arange(0, 2 * np.pi, np.pi / 100)


def bpsk(bitString):
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