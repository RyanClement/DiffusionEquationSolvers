#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffEq1D_ForwardEuler
Created: Aug 2020
@author: Ryan Clement (RRCC)
         scisoft@outlook.com
"""


### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


### FUNCTIONS
def plotComputedSteadyStateSolution():
    t = t0
    stop = 1
    tol  = 1e-5
    while stop:
        maxDif = 0.0
        # Advance solution one time step
        for i in range(1,xPts-1):
            u[i] = uO[i] + dp*(uO[i-1] -2.0*uO[i] + uO[i+1])
            dif  = abs(u[i] - uO[i])
            if maxDif < dif:
                maxDif = dif
        # Enforce boundary conditions
        u[0]  = lBC
        u[-1] = rBC
        if maxDif > tol:
            # Update previous solution
            uO[:] = u
            # Update simulation time
            t += dt
        else:
            stop = 0
    # Plot solution
    plt.plot(x,u,color='red')
    plt.title("Computed Steady State Solution")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.grid(True)
    plt.text(0.62,8.2,r'$u_t=\alpha\cdot u_{xx}$')

def initAnim():
    solPlt.set_data(x,uO)
    return solPlt,

def animate(n):
    for i in range(1,xPts-1):
        u[i] = uO[i] + dp*(uO[i-1] -2.0*uO[i] + uO[i+1])
    # Enforce boundary conditions
    u[0]  = lBC
    u[-1] = rBC
    uO[:] = u
    solPlt.set_data(x,u)
    return solPlt,




if __name__ == '__main__':
    ### VARIABLES
    x0   = 0.0                        # Minumum x-position (left boundary)
    xN   = 1.0                        # Maximum x-position (right boundary)
    xPts = 21                         # Number of spatial mesh points
    t0   = 0.0                        # Simulation start time
    dc   = 0.01                       # Diffusion coefficient
    dx   = (xN - x0)/(xPts - 1)       # Spatial mesh interval (distance betwen grid points)
    dt   = dx**2/(4*dc)               # Temporal mesh interval (time between solution updates)
    dp   = dc*dt/dx**2                # Dimensionless parameter
    x    = np.linspace(x0,xN,xPts)    # Spatial mesh array
    u    = np.zeros(xPts)             # Updated solution array
    uO   = np.zeros(xPts)             # Previous time-step value array
    lBC  = 10.0                       # Left boundary condition
    rBC  = 0.0                        # Right boundary condition
    # Set initial conditions
    uO[0]  = lBC
    uO[-1] = rBC

    # Option 1: Plot Computed Steady State Solution
    # Option 2: Movie
    option = 2
    if 1 == option:
        plotComputedSteadyStateSolution()
    else:
        fig, ax = plt.subplots()
        ax.set_title(r'$u_t=\alpha\cdot u_{xx}$')
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_xlim(x0,xN)
        ax.set_ylim(rBC,lBC)
        ax.grid(True)
        solPlt, = ax.plot([],[],color='red')
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames = 100,
                                       interval = 100,
                                       repeat_delay = 1000)
    plt.show()
