#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:36:09 2016

@author: ilker bayram
"""
import numpy as np
import matplotlib.pyplot as plt

def STFT(x,win,hop):
    # input variables :
    # x : audio signal in the time domain
    # win : window to be used for the STFT
    # hop : hop-size
    #
    # output variables : 
    # X : the STFT coefficients of x
    Lx = x.size # length of the audio signal
    Lwin = win.size #length of the window -- fft size 
    Kstep = (np.ceil((Lx-Lwin)/hop) + 1).astype(int)# number of steps to take
    X = np.zeros((Lwin,Kstep), dtype = complex ) # will hold the STFT coefficients
    k = 0 # time index
    nf = np.sqrt(Lwin) # normalizing factor
    for k in range(Kstep-1):
        d = x[ k * hop : k * hop + Lwin ] * win
        X[:,k] = np.fft.fft(d) / nf
    # the last window may partially overlap with the signal
    d = x[ Kstep * hop : ]
    X[:,k] = np.fft.fft(d * win[:d.size], n = Lwin) / nf
    return X
    
def ISTFT(X,win,hop):
    # input variables :
    # X : STFT coefficients
    # win : window to be used for the STFT
    # hop : hop-size
    #
    # output variables : 
    # x : inverse STFT of X
    Lwin = X.shape[0] # length of the window
    Kstep = X.shape[1] # number of frames
    Lx = Lwin + (Kstep - 1) * hop # length of the output signal
    x = np.zeros((Lx), dtype = complex)
    k = 0
    fac = np.sqrt(Lwin) # normalizing factor
    for k in range(Kstep):
        d = np.fft.ifft(X[:,k])
        d = fac * d * win
        x[ k * hop : k * hop + Lwin] += d

    return x
    
def NormalizeWindow(win,hop):
    # normalize the window according to the 
    # provided hop-size so that the STFT is a tight frame
    N = win.size
    K = int(N / hop)
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
    win2 = win / np.sqrt(z)
    return win2
    
def DisplaySTFT(X,fs,hop,Fr,clim):
    # input variables :
    # X : STFT coefficients
    # fs : sampling frequency in Hz (assumed to be integer)
    # hop : hop-size used in the STFT (for labeling the time axis)
    # Fr : frequency range (in Hz) to display (length two array)
    # clim : illumination range
    N = X.shape[0]
    Fd = (Fr * N / fs).astype(int)
    Z = X[Fd[1]:Fd[0]:-1,:]
    Z = np.clip(np.log(np.abs(Z) + 1e-50),clim[0],clim[1])
    Z = 255 * (Z - clim[0]) / (clim[1] - clim[0])
    time = float(hop) / float(fs) * float(X.shape[1])
    plt.imshow(Z,extent=[0,time,Fr[0] / 1000,Fr[1] / 1000],aspect = "auto") 
    plt.ylabel('Frequency (Khz)')
    plt.xlabel('Time (sec)')
