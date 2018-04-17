"""
TEST fft gradient
for large N: good 
for large Np i.e. truely periodic signal: perfect
for small N: wiggles; for even N: imaginary part in y_gradient_fft

4. for even NX, np.fft.ifft(1j*kb*fft(b)) is real 
"""

import numpy as np
from pylab import *

N = 100
Np = 1
y = np.sin(np.arange(N)/1./N * np.pi * Np)

yf = np.fft.fft(y)

k = np.fft.fftfreq(N)

y_gradient_fft = np.fft.ifft(yf * k * 1j * 2. * np.pi)
y_gradient_config = np.gradient(y)

figure()
plot(np.arange(N), y_gradient_config, 'b--', linewidth = 2, label='config')
plot(np.arange(N), y_gradient_fft, 'k-', label='fft')
legend()
show()





