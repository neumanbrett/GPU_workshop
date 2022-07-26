# Implement Cavity Flow with Navier-Stokes

from turtle import st
import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import time
 

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
    def __init__(self, dims, timesteps, use_gpu=False):
        # Parameter Initialization
        self.n = dims
        self.nt = timesteps
        self.use_gpu = use_gpu
        self.nit = 50
        self.c = 1
        self.dx = 2 / (self.n-1)
        self.dy = 2 / (self.n-1)
        self.rho = 1
        self.nu = 0.1
        self.dt = 0.001
        
        self._init_params(self.n)


    def _init_params(self, n):
        # CuPy setup
        self.x = cp.linspace(0, 2, n, dtype=cp.double)
        self.y = cp.linspace(0, 2, n, dtype=cp.double)
        self.X, self.Y = cp.meshgrid(self.x, self.y)
        self.u = cp.zeros((n, n), dtype=cp.double)
        self.v = cp.zeros((n, n), dtype=cp.double)
        self.p = cp.zeros((n, n), dtype=cp.double)         
        self.un = cp.empty_like(self.u, dtype=cp.double)
        self.vn = cp.empty_like(self.v, dtype=cp.double)
        self.pn = cp.empty_like(self.p, dtype=cp.double)
        self.b = cp.zeros((n, n), dtype=cp.double)


    def plot(self):
        fig = pyplot.figure(figsize=(11,7), dpi=100)
        # plotting the pressure field as a contour
        pyplot.contourf(self.X, self.Y, self.p, alpha=0.5, cmap=cm.viridis)  
        pyplot.colorbar()
        # plotting the pressure field outlines
        pyplot.contour(self.X, self.Y, self.p, cmap=cm.viridis)  
        # plotting velocity field
        pyplot.quiver(self.X[::2, ::2], self.Y[::2, ::2], self.u[::2, ::2], self.v[::2, ::2]) 
        pyplot.xlabel('X')
        pyplot.ylabel('Y')
        pyplot.title('Cavity Flow (NT = %i)' % self.nt)


    def _build_up_b(self): 
        self.b[1:-1, 1:-1] = (cp.multiply(self.rho, (cp.divide(1, self.dt)) * 
                             ((cp.subtract(self.u[1:-1, 2:], self.u[1:-1, 0:-2]) / 
                             (cp.add(cp.multiply(2, self.dx), cp.subtract((self.v[2:, 1:-1], cp.divide(self.v[0:-2, 1:-1]), cp.multiply(2, self.dy)) -
                             (cp.divide(cp.subtract(self.u[1:-1, 2:], self.u[1:-1, 0:-2]), cp.square(cp.multiply(2, self.dx))) -
                             cp.multiply(2, (cp.divide(cp.subtract(self.u[2:, 1:-1], self.u[0:-2, 1:-1]), cp.multiply(2, self.dy))) *
                             (cp.divide(cp.subtract(self.v[1:-1, 2:], self.v[1:-1, 0:-2]), cp.multiply(2, self.dx)))-
                             (cp.divide(cp.subtract(self.v[2:, 1:-1], self.v[0:-2, 1:-1]), cp.square(cp.multiply(2, self.dy)))))


    def _pressure_poisson(self):
        self.pn = self.p.copy()
        
        for q in range(self.nit):
            self.pn = self.p.copy()
            self.p[1:-1, 1:-1] = (((self.pn[1:-1, 2:] + self.pn[1:-1, 0:-2]) * self.dy**2 + 
                                 (self.pn[2:, 1:-1] + self.pn[0:-2, 1:-1]) * self.dx**2) /
                                 (2 * (self.dx**2 + self.dy**2)) -
                                 self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * 
                                 self.b[1:-1,1:-1])

            # dp/dx = 0 at x = 2
            self.p[:, -1] = self.p[:, -2]
            # dp/dy = 0 at y = 0
            self.p[0, :] = self.p[1, :]
            # dp/dx = 0 at x = 0
            self.p[:, 0] = self.p[:, 1]
            # p = 0 at y = 2 
            self.p[-1, :] = 0


    def compute(self):
        #if (self.use_gpu):
        #    start_gpu = cp.cuda.Event()
        #    start_gpu.record()
        
        for n in range(self.nt):
            un = self.u.copy()
            vn = self.v.copy()
            
            self._build_up_b()
            self._pressure_poisson()
            
            self.u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                             un[1:-1, 1:-1] * self.dt / self.dx *
                            (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * self.dt / self.dy *
                            (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                             self.dt / (2 * self.rho * self.dx) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
                             self.nu * (self.dt / self.dx**2 *
                            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                             self.dt / self.dy**2 *
                            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

            self.v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                             un[1:-1, 1:-1] * self.dt / self.dx *
                            (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                             vn[1:-1, 1:-1] * self.dt / self.dy *
                            (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                             self.dt / (2 * self.rho * self.dy) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
                             self.nu * (self.dt / self.dx**2 *
                            (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                             self.dt / self.dy**2 *
                            (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

            self.u[0, :]  = 0
            self.u[:, 0]  = 0
            self.u[:, -1] = 0
            # Set velocity on cavity lid equal to 1
            self.u[-1, :] = 1
            self.v[0, :]  = 0
            self.v[-1, :] = 0
            self.v[:, 0]  = 0
            self.v[:, -1] = 0
        
        #if (self.use_gpu):
        #    end_gpu = cp.cuda.Event()
        #    end_gpu.record()
        #    end_gpu.synchronize()


def launch_test(n, ts ):
    flow = CavityFlow(n, ts)
    flow.compute()
        

if __name__ == "__main__":
    # 128x128 Grid, 500 Timesteps
    n = 128
    ts = 500
    print(benchmark(launch_test, (n, ts), n_repeat=20, n_warmup=5))