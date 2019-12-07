#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import jit, cuda 



# In[2]:


time = np.arange(0, 2*np.pi, np.pi/100)


# In[3]:


# @jit(target ="cuda")
def bpsk(binString):
    amplitude = []
    for i in range(len(binString)):
        if binString[i] == '1': 
            amplitude.append(np.sin(time))
        else : 
            amplitude.append(-np.sin(time))
    return amplitude


# In[4]:


# @jit(target ="cuda")
def bpsk_demod(r_bb,L):
    x=np.real(r_bb)
    x = np.convolve(x,np.ones(L))
    x = x[np.arange(L,len(x),L)]
    ak_cap = (x > 0)
    return ['1' if x else '0' for x in ak_cap]


# In[5]:


# @jit(target ="cuda")
def bask(binString):
    amplitude = []
    for i in range(len(binString)):
        time = np.arange(0, 2*np.pi, np.pi/100)
        if binString[i] == '1': 
            amplitude.append(np.sin(time))
        else : 
            amplitude.append(np.zeros(time.size))
    return amplitude


# In[6]:


# @jit(target ="cuda")
def awgn(sinal, regsnr):
    sigpower=sum([np.power(abs(sinal[i]),2) for i in range(len(sinal))])
    sigpower=sigpower/len(sinal)
    noisepower=sigpower/(np.power(10,regsnr/10))
    noise=np.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(sinal)))
    return noise


# In[7]:


# @jit(target ="cuda")
def ber(binString, demodBinString):
    return sum(1 for a, b in zip(binString, demodBinString) if a != b) + abs(len(binString) - len(demodBinString))


# In[8]:


# @jit(target ="cuda")
# @cuda.jit
def simulate():
    pix = cv2.imread('photo1.jpg')
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
    cv2.imwrite("photo_changed_1.jpg", demod_img)

    img = mpimg.imread('photo1.jpg')
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


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
simulate()


# In[ ]:





# In[ ]:




