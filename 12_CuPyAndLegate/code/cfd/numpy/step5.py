# Implementing 2D linear convection

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sys


if __name__ == "__main__":
    # Parameter Init
    nx = 81
    ny = 81
    nt = 100
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.2
    dt = dx * sigma

    # NumPy setup
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx)) # 1xN vector of 1's
    un = np.ones((ny, nx))

    # Initial Conditions
    u[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2

    # Plot Initial Conditions
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

    # Running 2D
    u = np.ones((ny, nx))
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    ## Loop across number of time steps
    for n in range(nt + 1): 
        un = u.copy()
        row, col = u.shape
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                    (c * dt / dy * (un[j, i] - un[j - 1, i])))
                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1

    # Plotting 2D
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)

    # Plotting 2D with Array Functions
    u = np.ones((ny, nx))
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    ## Loop across number of time steps
    for n in range(nt + 1): 
        un = u.copy()
        u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                                (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
