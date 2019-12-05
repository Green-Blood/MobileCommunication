import numpy as np
import matplotlib.pylab as plt

image = plt.imread("Peruza.jpg")

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

# Function to plot RGB version of image, each pixel is a three integers, we just need to split them
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

# Bpsk Modulation
def bpskModulation(bitString):
    import math
    for i in range(len(bitString)):
        time = np.arange(0, 2 * math.pi, math.pi / 100 )
        if bitString[i] == '1':
            amplitude = np.sin(time)
        else:
            amplitude = np.sin(time)
            plt.plot(time,amplitude)
            plt.show()
    return



import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

time = np.arange(0, 2*np.pi, np.pi/100)

    def bpsk(binString):
        amplitude = []
        for i in range(len(binString)):
            if binString[i] == '1':
                amplitude.append(np.sin(time))
            else :
                amplitude.append(-np.sin(time))
        return amplitude

    def bpsk_demod(r_bb,L):
        x = np.real(r_bb)
        x = np.convolve(x,np.ones(L))
        x = x[np.arange(L,len(x),L)]
        ak_cap = (x > 0)
        return ['1' if x else '0' for x in ak_cap]

    def bask(binString):
        amplitude = []
        for i in range(len(binString)):
            time = np.arange(0, 2*np.pi, np.pi/100)
            if binString[i] == '1':
                amplitude.append(np.sin(time))
            else :
                amplitude.append(np.zeros(time.size))
        return amplitude
        sigpower=sum([np.power(abs(sinal[i]),2) for i in range(len(sinal))])
        sigpower=sigpower/len(sinal)
        noisepower=sigpower/(np.power(10,regsnr/10))
        noise=np.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(sinal)))
        return noise

    def ber(binString, demodBinString):
        return sum(1 for a, b in zip(binString, demodBinString) if a != b) + abs(len(binString) - len(demodBinString))

    def simulate():
        pix = cv2.imread('Peruza.jpg')
        height, width, channels = pix.shape
        demod_img = np.zeros((height, width, channels))
        BER = 0
        #     print('______________________________________________________________________________________________________________')
        for i in range(height):
            for j in range(width):
                rgb=pix[i][j]
                binString=''
                for k in np.arange(3):
                    binary = bin(rgb[k])[2:].zfill(8)
                    binString += binary
        #             print(binString)
                signal = np.array(bpsk(binString)).flatten()
        #             %matplotlib inline
        #             plt.figure()
        #             plt.plot(signal)
        #             plt.show()
                SNR = 1000
                nsignal = awgn(signal, SNR)
        #             %matplotlib inline
        #             plt.figure()
        #             plt.plot(nsignal+signal)
        #             plt.show()
                demodBinString = ''.join(map(str, bpsk_demod( nsignal+signal, 200)));
        #             print(demodBinString)
        #             print()
                demod_img[i][j] = [int(demodBinString[0:8],2),int(demodBinString[8:16],2),int(demodBinString[16:24],2)]
                BER += ber(binString,demodBinString)
        #             print('______________________________________________________________________________________________________________')
        print(BER/(height*width*channels*8))
        cv2.imwrite("photo_changed.jpg\", demod_img")

        img = mpimg.imread('photo2.jpg')
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(img)
        a.set_title('Before')
        plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

        img = mpimg.imread('photo_changed.jpg')
        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(img)
        imgplot.set_clim(0.0, 0.7)
        a.set_title('After')
        plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')

        return

    simulate()






