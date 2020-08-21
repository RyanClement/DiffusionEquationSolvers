#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionEq1D_ForwardEulerV
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

This program is very similar to the diffusionEq1D_ForwardEuler.py simulation. The
main two differences are vectorization and an additional BC/IC combination.
"""


### IMPORTS
import numpy as np
import matplotlib.pyplot as plt


### FUNCTIONS
def plotComputedSteadyStateSolution(u,uO):
    """
    Plots computed steady state solution.

    Parameters
    ----------
    u : numpy array
        Current solution
    uO: numpy array
        Old solution

    Returns
    -------
    None.

    """
    t = 0
    stop = 1
    tol  = 1e-5
    while stop:
        # Advance 10 time steps
        for j in range(10):
            u[1:xPtsM1] = uO[1:xPtsM1] + dp*(uO[0:xPtsM1-1] -2.0*uO[1:xPtsM1] + uO[2:xPts+1])
            # Enforce boundary conditions
            u[0]  = lBC
            u[-1] = rBC
            uO, u = u, uO
            t += dt
        # Check solution against tolerance
        maxDif  = np.amax( np.abs(u - uO) )
        if maxDif < tol:
            t -= dt
            stop = 0
    print("\nSolution met convergence criterion at time = "+str(t)+" s")
    # Plot solution
    plt.plot(x,u,color='blue')
    plt.title("Computed Steady State Solution")
    plt.ylabel("u")
    plt.xlabel("x")
    plt.grid(True)
    plt.text(0.62,8.2,r'$u_t=\alpha\cdot u_{xx}$')
    plt.show()

def plotComputedSolution(n,u,uO):
    """
    Plot solution at tStop timesteps.

    Parameters
    ----------
    n: Integer
        Advance solution n time steps
    u : Numpy array
        Current solution
    uO: Numpy array
        Old solution

    Returns
    -------
    None.

    """
    for j in range(n):
        u[1:xPtsM1] = uO[1:xPtsM1] + dp*(uO[0:xPtsM1-1] -2.0*uO[1:xPtsM1] + uO[2:xPts+1])
        # Enforce boundary conditions
        u[0]  = lBC
        u[-1] = rBC
        uO, u = u, uO
    t = n*dt
    s = "\nSolution at time = "+str(t)+" s"
    # Plot solution
    plt.plot(x,u,color='purple')
    plt.title(s)
    plt.ylabel("u")
    plt.xlabel("x")
    plt.grid(True)
    plt.text(0.62,8.2,r'$u_t=\alpha\cdot u_{xx}$')
    plt.show()



if __name__ == '__main__':
    ### VARIABLES
    x0   = 0.0                        # Minumum x-position (left boundary)
    xN   = 1.0                        # Maximum x-position (right boundary)
    xPts = 101                        # Number of spatial mesh points
    xPtsM1 = xPts - 1
    t0   = 0.0                        # Simulation start time
    dc   = 0.01                       # Diffusion coefficient
    dx   = (xN - x0)/xPtsM1           # Spatial mesh interval (distance betwen grid points)
    dt   = dx**2/(4*dc)               # Temporal mesh interval (time between solution updates)
    dp   = dc*dt/dx**2                # Dimensionless parameter
    x    = np.linspace(x0,xN,xPts)    # Spatial mesh array
    u    = np.zeros(xPts)             # Updated solution array
    uO   = np.zeros(xPts)             # Previous time-step value array
    lBC  = 10.0                       # Left boundary condition
    rBC  = 0.0                        # Right boundary condition

    plotComputedSteadyStateSolution(u,uO)
    u.fill(0)
    uO.fill(0)
    plotComputedSolution(1000,u,uO)

