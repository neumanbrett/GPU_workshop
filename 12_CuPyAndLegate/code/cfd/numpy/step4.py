# Implementing Burgers' Equation
# Combining nonlinear convection and diffusion
# Ignoring the SymPy sections and implemented function

import numpy as np
from matplotlib import pyplot
import time
import sys


def ufunc(t, x, nu):
    return -2*nu*(-(-8*t + 2*x)*np.exp(-(-4*t + x)**2/(4*nu*(t + 1)))
        /(4*nu*(t + 1)) - (-8*t + 2*x - 4*np.pi)*np.exp(-(-4*t + x - 2*np.pi)**2
        /(4*nu*(t + 1)))/(4*nu*(t + 1)))/(np.exp(-(-4*t + x - 2*np.pi)**2
        /(4*nu*(t + 1))) + np.exp(-(-4*t + x)**2/(4*nu*(t + 1)))) + 4

if __name__ == "__main__":
    nx = 101
    dx = 2 * np.pi / (nx - 1)
    nt = 100
    nu = 0.07 # viscosity value
    dt = dx * nu

    x = np.linspace(0, 2 * np.pi, nx)
    un = np.empty(nx)
    t = 0

    u = np.asarray([ufunc(t, x0, nu) for x0 in x])
    print(u)

    for n in range(nt):
        un = u.copy()
        for i in range(1, nx-1):
            u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
        
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
        
        u[-1] = u[0]
        
    u_analytical = np.asarray([ufunc(nt * dt, xi, nu) for xi in x])

    pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.plot(x,u, marker='o', lw=2, label='Computational')
    pyplot.plot(x, u_analytical, label='Analytical')
    pyplot.xlim([0, 2 * np.pi])
    pyplot.ylim([0, 10])
    pyplot.legend();