# Implementing 2D linear convection

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import timeit
import sys


if __name__ == "__main__":
    # Parameter Init
    nx = 101
    ny = 101
    nt = 80
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.2
    dt = dx * sigma

    # NumPy setup
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx)) # 1xN vector of 1's
    v = np.ones((ny, nx))
    un = np.ones((ny, nx))
    vn = np.ones((ny, nx))

    # Initial Conditions
    u[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2
    v[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2

    # Plotting 2D with Array Functions
    u = np.ones((ny, nx))
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    # Using the standard array method of computing
    u = np.ones((ny, nx))
    u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    # Looping over time steps
    for n in range(nt + 1):
        un = u.copy()
        vn = v.copy()

        u[1:, 1:] = (un[1:, 1:] -
                     (un[1:, 1:] * c * dt / dx *(un[1:, 1:] - un[1:, :-1])) -
                      vn[1:, 1:] * c * dt / dy *(un[1:, 1:] - un[:-1, 1:]))
        v[1:, 1:] = (vn[1:, 1:] -
                     (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) -
                     vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')