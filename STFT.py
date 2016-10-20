#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:36:09 2016

@author: ilker bayram
"""
import numpy
import scipy
import matplotlib.pyplot as plt

def STFT(x,win,hop):
    # input variables :
    # x : audio signal in the time domain
    # win : window to be used for the STFT
    # hop : hop-size
    #
    # output variables : 
    # X : the STFT coefficients of x
    L = len(x) # length of the audio signal
    N = len(win) #length of the window -- fft size 
    K = (L-N)/hop + 1 # number of steps to take
    X = numpy.zeros((N,K), dtype = complex ) # will hold the STFT coefficients
    k = 0 # time index
    nf = numpy.sqrt(N) # normalizing factor
    while (k < K):
        d = x[ k * hop : k * hop + N ] * win
        D = scipy.fft(d)
        X[:,k] = D / nf
        k += 1
    return X
    
def ISTFT(X,win,hop):
    # input variables :
    # X : STFT coefficients
    # win : window to be used for the STFT
    # hop : hop-size
    #
    # output variables : 
    # x : inverse STFT of X
    N = X.shape[0] # length of the window
    K = X.shape[1] # number of frames
    L = N + (K-1) * hop # length of the output signal
    x = numpy.zeros((L), dtype = complex)
    k = 0
    fac = numpy.sqrt(N) # normalizing factor
    while (k < K):
        d = scipy.ifft(X[:,k])
        d = fac * d * win
        x[k*hop:k*hop+N] += d
        k += 1
    return x
    
def NormalizeWindow(win,hop):
    # normalize the window according to the 
    # provided hop-size so that the STFT is a tight frame
    N = len(win)
    K = N / hop
    win2 = win * win
    z = 1 * win2
    k = 1
    ind1 = N - hop
    ind2 = hop
    while (k < K):
        z[0:ind1] += win2[ind2:N]
        z[ind2:N] += win2[0:ind1]
        ind1 -= hop
        ind2 += hop
        k += 1
    win2 = win / numpy.sqrt(z)
    return win2
    
def DisplaySTFT(X,fs,hop,Fr,clim):
    # input variables :
    # X : STFT coefficients
    # fs : sampling frequency in Hz (assumed to be integer)
    # hop : hop-size used in the STFT (for labeling the time axis)
    # Fr : frequency range (in Hz) to display (length two array)
    # clim : illumination range
    N = X.shape[0]
    Fd = Fr * N / fs
    Z = X[Fd[1]:Fd[0]:-1,:]
    Z = numpy.clip(numpy.log(numpy.abs(Z)),clim[0],clim[1])
    Z = 255 * (Z - clim[0]) / (clim[1] - clim[0])
    time = float(hop) / float(fs) * float(X.shape[1])
    plt.imshow(Z,extent=[0,time,Fr[0] / 1000,Fr[1] / 1000],aspect = "auto") 
    plt.ylabel('Frequency (Khz)')
    plt.xlabel('Time (sec)')