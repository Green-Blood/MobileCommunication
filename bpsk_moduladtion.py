import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

src = 'Peruza.jpg'
SNR_dbs = [-10, -5, 0, 5, 10]
BerBPSK =[]
Fs = 12
f = 1
sample = 12
x = np.arange(sample)
sin = np.sin(2 * np.pi * f * x / Fs)
rsin = -sin
size = 100

def resizer(src):
    img = cv.imread(src, cv.IMREAD_UNCHANGED)
    print("Original shape : ", img.shape)
    if img.shape[0] >= img.shape[1]:
        ratio = img.shape[1] / img.shape[0]
        height = size
        width = int(ratio * height)
        dim = (width, height)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    else:
        ratio = img.shape[0] / img.shape[1]
        width = size
        height = int(ratio * width)
        dim = (width, height)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    print("img resized: ", resized.shape)

    return resized


def RGB_bit(img):
    red = np.array(img[:, :, 2].flat)
    green = np.array(img[:, :, 1].flat)
    blue = np.array(img[:, :, 0].flat)
    bin_string = ''
    inline_rgb = np.concatenate((red, green, blue), axis=None)
    for c in np.nditer(inline_rgb):
        bin_string += bin(c)[2:].rjust(8, "0")
    bit_string = np.array([int(x, 2) for x in bin_string])
    return bit_string


def sqrt_SNR(SNR_db):
        return math.sqrt(0.5 * 10 ** (-SNR_db / 10))


def signal_generator(bit_length, bit_string):
    # create signal
    signal = np.zeros((bit_length,12), float)
    for i in range(bit_length):
        if bit_string[i] > 0:
            signal[i] = sin
        else:
            signal[i] = rsin
    return signal


def received_signal(bit_length, signal, SNR):
    # create noise with AWGN
    noise = np.random.randn(bit_length, 12)
    received_signal = signal + noise * SNR
    return received_signal


def decode_signal(r_signal, bit_length):
    rgb_bit = np.zeros(bit_length, int)
    for i in range(bit_length):
        sum1 = sum([np.power(j,2) for j in np.nditer(r_signal[i]-sin)])
        sum2 = sum([np.power(j,2) for j in np.nditer(r_signal[i]-rsin)])
        if sum1 < sum2:
            rgb_bit[i] = 1
        else:
            rgb_bit[i] = 0
    return rgb_bit


def error(signal, r_signal):
    error_matrix = abs((r_signal - signal) / 2)
    error = error_matrix.sum()
    return error


def decode_img(signal, width, heigth):
    size_chanel = width * heigth * 8
    bin_string = ''.join(str(color) for color in np.nditer(signal))
    r, g, b = [], [], []
    for i in range(0, size_chanel, 8):
        r.append(int(bin_string[i:i + 8], 2))
        g.append(int(bin_string[(size_chanel + i):(size_chanel + i + 8)], 2))
        b.append(int(bin_string[(2 * size_chanel + i):(2 * size_chanel + i + 8)], 2))
    red = np.array(r, int)
    green = np.array(g, int)
    blue = np.array(b, int)
    red = np.reshape(red, (heigth, width))
    green = np.reshape(green, (heigth, width))
    blue = np.reshape(blue, (heigth, width))
    img = np.dstack((blue, green, red))
    return img


if __name__ == '__main__':
    img = resizer(src)
    height = img.shape[0]
    width = img.shape[1]
    bit_string = RGB_bit(img)
    bit_length = len(bit_string)

    signal = signal_generator(bit_length, bit_string)
    for SNR_db in SNR_dbs:
        SNR_val = sqrt_SNR(SNR_db)
        r_signal = received_signal(bit_length, signal, SNR_val)
        modulated_signal = decode_signal(r_signal, bit_length)
        BerBPSK.append(error(bit_string, modulated_signal)/bit_length)
        print(BerBPSK)
        result = decode_img(modulated_signal, width, height)
        fname = 'recieved_imgSNRdb='+str(SNR_db)+'.jpg'
        cv.imwrite(fname, result)

    plt.semilogy(SNR_dbs, BerBPSK, '-')
    plt.ylabel('BER')
    plt.xlabel('SNR')
    plt.title('BPSK BER Curves')
    plt.legend('Simulation', loc='upper right')
    plt.grid()
    plt.show()
