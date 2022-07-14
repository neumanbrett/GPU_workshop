# Implementing the nonlinear convection algorithm

import numpy as np
from matplotlib import pyplot
import time
import sys


if __name__ == "__main__":
    nx = 41
    dx = 2 / (nx - 1)
    nt = 25
    dt = 0.025

    u = np.ones(nx)
    u[int(.5 / dx):int(1 / dx + 1)] = 2

    un = np.ones(nx)

    for n in range(nt):
        un = u.copy() # Copy existing values to new numpy array
        for i in range(1, nx):
            # nonlinear convection, no longer uses constant c
            u[i] = un[i] - un[i] * dt/dx * (un[i] - un[i-1])

pyplot.plot(np.linspace(0, 2, nx), u);