# Implement Channel Flow with Navier-Stokes

import numpy as np
import cupy as cp
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sys


def build_up_b(rho, dt, dx, dy, u, v):
    xp = cp.get_array_module(x) 
    b = xp.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
                    + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))
                    - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2
                    * 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
                    - (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))
                    - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    
    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx)
                  + (v[2:, -1] - v[0:-2, -1]) / (2 * dy))
                  - ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2
                  - 2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy)
                  * (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx))
                  - ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx)
                 + (v[2:, 0] - v[0:-2, 0]) / (2 * dy))
                 - ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2
                 - 2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy)
                 - (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))
                 - ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))
    
    return b


def pressure_poisson_periodic(p, dx, dy):
    xp = cp.get_array_module(x) 
    pn = xp.empty_like(p)
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
                        + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2)
                        / (2 * (dx**2 + dy**2))
                        - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2
                      + (pn[2:, -1] + pn[0:-2, -1]) * dx**2)
                      / (2 * (dx**2 + dy**2))
                      - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2
                     +  (pn[2:, 0] + pn[0:-2, 0]) * dx**2)
                     / (2 * (dx**2 + dy**2))
                     - dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])
        
        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
    
    return p


def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    xp = cp.get_array_module(x) 
    un = xp.empty_like(u)
    vn = xp.empty_like(v)
    b = xp.zeros((ny, nx))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson_periodic(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1]
                        - un[1:-1, 1:-1] * dt / dx
                        * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
                        - vn[1:-1, 1:-1] * dt / dy
                        * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
                        - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
                        + nu * (dt / dx**2
                        * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                        + dt / dy**2
                        * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                       - un[1:-1, 1:-1] * dt / dx
                       * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
                       - vn[1:-1, 1:-1] * dt / dy
                       * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
                       - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
                       + nu * (dt / dx**2
                       * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                       + dt / dy**2
                       * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

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
    nx = 41
    ny = 41
    nt = 10
    nit = 50
    c = 1
    dx = 2 / (nx-1)
    dy = 2 / (ny-1)
    rho = 1
    nu = 0.1
    F = 1
    dt = 0.01

    # CuPy setup
    xp = cp.get_array_module(main)
    
    x = xp.linspace(0, 2, nx)
    y = xp.linspace(0, 2, ny)
    X, Y = xp.meshgrid(x, y)

    # Initial Conditions
    u = xp.zeros((ny, nx))
    un = xp.zeros((ny, nx))
    v = xp.zeros((ny, nx))
    vn = xp.zeros((ny, nx))
    p = xp.zeros((ny, nx))
    pn = xp.zeros((ny, nx))
    b = xp.zeros((ny, nx))

    udiff = 1
    stepcount = 0

    while udiff > .001:
        un = u.copy()
        vn = v.copy()

        b = build_up_b(rho, dt, dx, dy, u, v)
        p = pressure_poisson_periodic(p, dx, dy)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx * 
                        (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy * 
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                        dt / (2 * rho * dx) * 
                        (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                        nu * (dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        dt / dy**2 * 
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) + 
                        F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx * 
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy * 
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * 
                        (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 * 
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2     
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx * 
                    (un[1:-1, -1] - un[1:-1, -2]) -
                    vn[1:-1, -1] * dt / dy * 
                    (un[1:-1, -1] - un[0:-2, -1]) -
                    dt / (2 * rho * dx) *
                    (p[1:-1, 0] - p[1:-1, -2]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                    dt / dy**2 * 
                    (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (un[1:-1, 0] - un[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy * 
                    (un[1:-1, 0] - un[0:-2, 0]) - 
                    dt / (2 * rho * dx) * 
                    (p[1:-1, 1] - p[1:-1, -1]) + 
                    nu * (dt / dx**2 * 
                    (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                    dt / dy**2 *
                    (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                    (vn[1:-1, -1] - vn[1:-1, -2]) - 
                    vn[1:-1, -1] * dt / dy *
                    (vn[1:-1, -1] - vn[0:-2, -1]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, -1] - p[0:-2, -1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                    dt / dy**2 *
                    (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                    (vn[1:-1, 0] - vn[1:-1, -1]) -
                    vn[1:-1, 0] * dt / dy *
                    (vn[1:-1, 0] - vn[0:-2, 0]) -
                    dt / (2 * rho * dy) * 
                    (p[2:, 0] - p[0:-2, 0]) +
                    nu * (dt / dx**2 * 
                    (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                    dt / dy**2 * 
                    (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))

        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :]=0
        
        udiff = (cp.sum(u) - cp.sum(un)) / cp.sum(u)
        stepcount += 1

    print(stepcount)

    fig = pyplot.figure(figsize = (11,7), dpi=100)
    pyplot.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])