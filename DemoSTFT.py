# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:45:04 2016

@author: ilker bayram
"""
# demonstrates the use of the routines in the STFT package

import numpy
import STFT
#import pdb
import pylab
import matplotlib.pyplot as plt
from scipy.io import wavfile

# load the observed audio
fs, x = wavfile.read('observed.wav')
x = pylab.double(x) / 2.**15 # convert integer to double

# STFT parameters
winms = 60 # length of the window in milliseconds
win = numpy.hamming(winms*fs/1000)  # we use a Hamming window but will modify it 

hopms = 15 # hop size in milliseconds
hop = hopms*fs/1000 # hop size in samples

win2 = STFT.NormalizeWindow(win,hop) # this normalization step ensures that the STFT is self-inverting (or a Parseval frame)

# Compute the STFT
X = STFT.STFT(x,win2,hop)

# Display the STFT
Fr = numpy.array([0,3000]) # Frequency range to display
clim = [-10, 0] # magnitude range in dB

Norm = numpy.amax(abs(X))  
X2 = X / Norm # normalize so that the largest STFT coefficient magnitude is unity

STFT.DisplaySTFT(X2,fs,hop,Fr,clim) # main display function for the STFT

# Inverse STFT
y = STFT.ISTFT(X,win2,hop)
yr = numpy.real(y)

# check that ISTFT actually inverts the STFT
t = numpy.arange(0,yr.size)
t = pylab.double(t) / pylab.double(fs)

# plot the original and the difference of the original from the reconstruction
plt.figure()
plt.plot(t,x[0:yr.size],'b-',label="Original")
plt.plot(t,x[0:yr.size]-yr,'r-',label = "Difference of the Reconstruction and the Original")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)
plt.xlabel('Time (sec)')