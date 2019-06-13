# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:04 2016

@author: ilker bayram
"""
# demonstrates the use of the routines in the STFT package

import numpy as np
import STFT
import pylab
import matplotlib.pyplot as plt
from scipy.io import wavfile

# load the observed audio
fs, x = wavfile.read('observed.wav')
x = pylab.double(x) / 2.**15 # convert integer to double

# STFT parameters
winms = 60 # length of the window in milliseconds
win = np.hamming(int(winms*fs/1000))  # we use a Hamming window but will modify it

hopms = 15 # hop size in milliseconds
hop = int(hopms*fs/1000) # hop size in samples

win2 = STFT.NormalizeWindow(win,hop) # this normalization step ensures that the STFT is self-inverting (or a Parseval frame)

# Compute the STFT
# zero pad to ensure that there are no partial overlap windows in the STFT computation
x = np.pad(x, (win.size + hop, win.size + hop), 'constant', constant_values=(0, 0))
X = STFT.STFT(x,win2,hop)

# Display the STFT
Fr = np.array([0,3000]) # Frequency range to display
clim = np.array([-10, 0]) # magnitude range in dB

Norm = np.amax(abs(X))  
X2 = X / Norm # normalize so that the largest STFT coefficient magnitude is unity
fig, ax = plt.subplots(1,1,figsize = (10,5))
STFT.DisplaySTFT(X2,fs,hop,Fr,clim) # main display function for the STFT

# Inverse STFT
y = STFT.ISTFT(X,win2,hop)
yr = np.real(y)

# check that ISTFT actually inverts the STFT
t = np.arange(0,x.size) / fs 

# plot the original and the difference of the original from the reconstruction
fig, ax = plt.subplots(1,1,figsize = (10,5))
ax.plot(t,x,'b-',label="Original")
ax.plot(t,x-yr[:x.size],'r-',label = "Difference of the Reconstruction and the Original")
ax.set_xlabel('Time (sec)')
ax.set_title('Maximum Absolutte Reconstruction Error : {:.4e}'.format(np.max(np.abs(x-yr[:x.size]))))
ax.legend()