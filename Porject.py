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

plotRGBImage()




