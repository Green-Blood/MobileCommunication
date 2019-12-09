import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


time = np.arange(0, 2*np.pi, np.pi/100)

def bpsk(bitString):
    amplitude = []
    for i in range(len(bitString)):
        if bitString[i] == '1': 
            amplitude.append(np.sin(time))
        else:
            amplitude.append(-np.sin(time))
    return amplitude

def plotImaging():
    
    return


def avgSigPow():


def add_GaussianNoise(analogS,SNRdb):
    # for index in range(len(analogS)):
    #     sigpower = np.power(abs(analogS[index]),2)
    # avgSignalPower = sum([sigpower])
    # noisepower=avgSignalPower(analogS)/SNRdb
    sigpower = sum([np.power(abs(analogS[i]),2) for i in range(len(analogS))])
    sigpower = sigpower/len(analogS)
    noisepower=sigpower/(np.power(10,SNRdb/10))
    noise=np.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(analogS)))
    return noise

    
def RGB(photo,channels,i,j):
    pixel=photo[i][j]
    bitString=''
    for k in range(channels):
        binary = bin(pixel[k])[2:].zfill(8)
        bitString += binary
    return bitString

if __name__ == "__main__":
    
    SNRdb = 10
    photo = cv2.imread('face.jpg')
    height, width, channels = photo.shape
    # print(channels)
    for i in range(height): 
        for j in range(width):
            bitString = RGB(photo,channels,i,j)
            analogSignal = np.array(bpsk(bitString)).flatten()
            noisySignal = GausianNoise(analogSignal,SNRdb)
            
    pass