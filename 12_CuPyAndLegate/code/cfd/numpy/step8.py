# Solve 2D Burgers' Equation
# Full convection nonlinearity and many known analytical solutions

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import timeit
import sys
    

if __name__ == "__main__":
    # Parameter Init
    nx = 41
    ny = 41
    nt = 120
    c = 1
    nu = 0.01
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.0009
    dt = dx * dy * sigma / nu

    # NumPy setup
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    un = np.ones((ny, nx))
    vn = np.ones((ny, nx))
    comb = np.ones((ny, nx))

    # Initial Conditions
    u[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2
    v[int(0.5 / dy):int(1 / dy + 1),int(0.5 / dx):int(1 / dx + 1)] = 2

    # Plotting Initial Conditions
    u = np.ones((ny, nx))
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis)
    ax.plot_surface(X, Y, v[:], rstride=1, cstride=1, cmap=cm.viridis)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    # Loop across number of time steps
    for n in range(nt + 1):
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        dt / dx * un[1:-1, 1:-1] * 
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                        dt / dy * vn[1:-1, 1:-1] * 
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + 
                        nu * dt / dx**2 * 
                        (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + 
                        nu * dt / dy**2 * 
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                        dt / dx * un[1:-1, 1:-1] *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        dt / dy * vn[1:-1, 1:-1] * 
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) + 
                        nu * dt / dx**2 * 
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        nu * dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
        
        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    # Plotting results
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.viridis)
    ax.plot_surface(X, Y, v, rstride=1, cstride=1, cmap=cm.viridis)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')