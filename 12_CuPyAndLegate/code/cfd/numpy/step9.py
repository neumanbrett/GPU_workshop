# Solve Laplace and Poisson equations for next steps

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import timeit
import sys
    

def plot2d(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


def laplace2d(p, y, dx, dy, l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)

    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2])
                          + dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1]))
                          / (2 * (dx**2 + dy**2)))
            
        p[:, 0] = 0  # p = 0 @ x = 0
        p[:, -1] = y  # p = y @ x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
        p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:]))
                  / np.sum(np.abs(pn[:])))
     
    return p


if __name__ == "__main__":
    # Parameter Init
    nx = 31
    ny = 31
    nt = 120
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)

    # NumPy setup
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    p = np.zeros((ny, nx))

    # Initial Conditions
    # p = 0 @ x = 0
    p[:,0] = 0
    # p = y @ x = 2
    p[:,-1] = y
    # dp/dy = 0 @ y = 0
    p[0,:] = p[1,:]
    # dp/dy = 0 @ y = 1
    p[-1,:] = p[-2,:]

    # Plotting Initial Conditions
    plot2d(x,y,p)

    # Plotting 2D Laplace
    p = laplace2d(p, y, dx, dy, 1e-4)
    plot2d(x, y, p)