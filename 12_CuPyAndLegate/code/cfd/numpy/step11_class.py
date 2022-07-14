# Implement Cavity Flow with Navier-Stokes

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import timeit
import sys
    
class CavityFlow(object):
    """ 
    Solve Cavity Flow

    Methods:
        constructor
        compute_flow
            build_b
            pressure_poisson
        plot

    Data:

    Usage:

    """
    def __init__(self, dims, timesteps):
        # Parameter Initialization
        self.n = args.n
        self.nt = args.nt
        self.nit = 50
        self.c = 1
        self.dx = 2 / (self.n-1)
        self.dy = 2 / (self.ny-1)
        self.rho = 1
        self.nu = 0.1
        self.dt = 0.001

        self.init_params(self.n)


    def _init_params(n):
        # NumPy setup
        x = np.linspace(0, 2, n)
        y = np.linspace(0, 2, n)
        X, Y = np.meshgrid(x, y)

        # Initial Conditions
        u = np.zeros((n, n))
        v = np.zeros((n, n))
        p = np.zeros((n, n))
        b = np.zeros((n, n))


    def plot(self, X, Y, p, u, v, nt):
        fig = pyplot.figure(figsize=(11,7), dpi=100)
        # plotting the pressure field as a contour
        pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
        pyplot.colorbar()
        # plotting the pressure field outlines
        pyplot.contour(X, Y, p, cmap=cm.viridis)  
        # plotting velocity field
        pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title('Cavity Flow (NT = %i)' % nt)


    def _build_up_b(self, b, rho, dt, u, v, dx, dy):
        
        b[1:-1, 1:-1] = (rho * (1 / dt * 
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                        (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                        ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                        2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                            (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

        return b


    def _pressure_poisson(self, p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()
        
        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                            (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) -
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                            b[1:-1,1:-1])

            # dp/dx = 0 at x = 2
            p[:, -1] = p[:, -2]
            # dp/dy = 0 at y = 0
            p[0, :] = p[1, :]
            # dp/dx = 0 at x = 0
            p[:, 0] = p[:, 1]
            # p = 0 at y = 2 
            p[-1, :] = 0
            
        return p


    def compute(self, nt, u, v, dt, dx, dy, p, rho, nu):
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))
        
        for n in range(nt):
            un = u.copy()
            vn = v.copy()
            
            b = self._build_up_b(b, rho, dt, u, v, dx, dy)
            p = self._pressure_poisson(p, dx, dy, b)
            
            u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                             un[1:-1, 1:-1] * dt / dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * dt / dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                             nu * (dt / dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                             dt / dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                            un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                            vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                            dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                            nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                            dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            u[0, :]  = 0
            u[:, 0]  = 0
            u[:, -1] = 0
            # Set velocity on cavity lid equal to 1
            u[-1, :] = 1
            v[0, :]  = 0
            v[-1, :] = 0
            v[:, 0]  = 0
            v[:, -1] = 0
            
        return u, v, p


if __name__ == "__main__":
    # Parameter Init
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        dest="nt",
        help="number of timesteps to run",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=41,
        dest="n",
        help="number of elements for X and Y dimensions",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        dest="device",
        help="set to run on cpu using numpy or gpu using cupy"
    )
    args, _ = parser.parse_known_args()

    nx = 41
    ny = 41
    nt = 500
    nit = 50
    c = 1
    dx = 2 / (nx-1)
    dy = 2 / (ny-1)
    rho = 1
    nu = 0.1
    dt = 0.001

    # Solving cavity flow
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # Plotting results for nt = 500
    plot2d(X, Y, p, u, v, nt)

    # Initializing for a new test with updated nt value
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    nt = 700
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    # Plot new data
    plot2d(X, Y, p, u, v, nt)