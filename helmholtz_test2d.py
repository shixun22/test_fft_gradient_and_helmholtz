"""
given 2D velocity field, decompose to solenoidal and compressive parts
for uniform grid with dx = dy = dz

Lessons learnt:
1. ifftn(fftn(f) * k) not real because k anti-symmetric
-- but ifftn(fftn(f) * ik) should be real
-- only when N is even, the Nyquist frequency is sampled only once, then the inverse transform is not real. but for smooth enough data (power at Nyquist frequency mild), just taking the real part of the result should not cause a problem
"""

import numpy as np
from pylab import *

NX = 21
NY = 41
Np = 1

Vfx = np.sin(np.arange(NX)/1./NX * np.pi * Np)[:,np.newaxis] * np.cos(np.arange(NY)/1./NY * np.pi * Np)[np.newaxis,:]

Vfy = np.sin(np.arange(NX)/1./NX * np.pi * Np * 3)[:,np.newaxis] * np.cos(np.arange(NY)/1./NY * np.pi * Np * 3)[np.newaxis,:]

vx_f = np.fft.fftn(Vfx)
vy_f = np.fft.fftn(Vfy)

kx = np.fft.fftfreq(NX).reshape(NX,1)
ky = np.fft.fftfreq(NY)
k2 = kx**2 + ky**2 
k2[0,0] = 1.

div_Vf_f = (kx * vx_f +  ky * vy_f) #* 1j
V_compressive_overk = div_Vf_f / k2
V_compressive_x = np.fft.ifftn(V_compressive_overk * kx) #[:,np.newaxis,np.newaxis])
V_compressive_y = np.fft.ifftn(V_compressive_overk * ky)

V_solenoidal_x = Vfx - V_compressive_x
V_solenoidal_y = Vfy - V_compressive_y

# check if the solenoidal part really divergence-free
divVs = np.fft.ifftn((np.fft.fftn(V_solenoidal_x) * kx + np.fft.fftn(V_solenoidal_y) * ky) * 1j)

X, Y = np.meshgrid(range(NY), range(NX))
figure()
quiver(X, Y, V_solenoidal_x, V_solenoidal_y)
figure()
quiver(X, Y, V_compressive_x, V_compressive_y)
ion()
show()

