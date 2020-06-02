# Basic-STFT
Code for basic STFT operations

Contains phython code to
1) normalize a window for to obtain a self-inverting STFT (or a Parseval frame),
2) realize STFT,
3) inverse STFT,
4) display the STFT coefficients

A demonstration is provided in DemoSTFT.py

The notebook titled 'GriffinLim' demonstrates the Griffin-Lim algorithm, reconstructing the speech signal from the magnitudes of its STFT coefficients.

A similar repository in Julia is available at https://github.com/ilkerbayram/STFT-Julia.
