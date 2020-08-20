#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionEq1D_ForwardEuler
Created: Aug 2020
@author: Ryan Clement (RRCC)
         scisoft@outlook.com

Purpose: Solve the

            u_t = alpha * u_xx

on the interval (0,L) with boundary conditions

            u = 10 for x = 0
            and
            u = 0 for x = 1.0

and initial condition

            u(x,0) = 0.0
"""


### IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


### FUNCTIONS
def plotComputedSteadyStateSolution():
    """
    Plots computed steady state solution.

    Returns
    -------
    None.

    """
    t = t0
    stop = 1
    tol  = 1e-5
    # Advance several time steps
    for j in range(3):
        for i in range(1,xPts-1):
            u[i] = uO[i] + dp*(uO[i-1] -2.0*uO[i] + uO[i+1])
        # Enforce boundary conditions
        u[0]  = lBC
        u[-1] = rBC
        uO[:] = u
    # Advance solution until tolerance is met.
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
    plt.plot(x,u,color='blue')
    plt.title("Computed Steady State Solution")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.grid(True)
    plt.text(0.62,8.2,r'$u_t=\alpha\cdot u_{xx}$')

def initAnim():
    """
    Initialization function for matplotlib.animation.FuncAnimation "init_func" argument.

    Returns
    -------
    solPlt : TYPE
        DESCRIPTION.

    """
    solPlt.set_data(x,uO)
    return solPlt,

def animate(n):
    """
    Animation function for matplotlib.animation.FuncAnimation "func" argument.

    Parameters
    ----------
    n : dummy
        Not used.

    Returns
    -------
    solPlt : axis.plt
        Plot object for animation.

    """
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
