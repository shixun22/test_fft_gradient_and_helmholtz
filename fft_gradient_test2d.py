"""
TEST fft gradient
for large N: good 
for large Np i.e. truely periodic signal: perfect
for small N: wiggles; for even N: imaginary part in y_gradient_fft

4. for even NX, np.fft.ifft(1j*kb*fft(b)) is real 
"""

import numpy as np
from pylab import *

NX = 11
NY = 21
Np = 1

y = np.sin(np.arange(NX)/1./NX * np.pi * Np)[:,np.newaxis] * np.cos(np.arange(NY)/1./NY * np.pi * Np)[np.newaxis,:]

yf = np.fft.fftn(y)

kx = np.fft.fftfreq(NX).reshape(NX,1)
ky = np.fft.fftfreq(NY)

y_gradient_fft = [np.fft.ifftn(yf * kx * 1j * 2. * np.pi), np.fft.ifftn(yf * ky * 1j * 2. * np.pi)]
y_gradient_config = np.gradient(y)

figure()
plot(np.arange(NX), y_gradient_config[0][:,2], 'b--', linewidth = 2, label='config')
plot(np.arange(NX), y_gradient_fft[0][:,2].real, 'k-', label='fft_real')
plot(np.arange(NX), y_gradient_fft[0][:,2].imag, 'm-', label='fft_imag')
legend()
show()





