# Implementing the Diffusion Equation in 1-D
# Dealing with the 2nd order derivative

import numpy as np
from matplotlib import pyplot
import time
import sys


if __name__ == "__main__":
    nx = 41
    dx = 2 / (nx - 1)
    nt = 25
    nu = 0.3 # viscosity value
    sigma = 0.2
    dt = sigma * dx**2 / nu

    u = np.ones(nx)
    u[int(.5 / dx):int(1 / dx + 1)] = 2

    un = np.ones(nx)

    for n in range(nt):
        un = u.copy() # Copy existing values to new numpy array
        for i in range(1, nx - 1):
            # nonlinear convection, no longer uses constant c
            u[i] = un[i] + nu * dt / dx**2 * (u[i+1] - 2 * u[i] + u[i-1])

pyplot.plot(np.linspace(0, 2, nx), u);