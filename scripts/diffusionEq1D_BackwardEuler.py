#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionEq1D_BackwardEuler
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
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt


def analSol(a,x,t):
    """


    Parameters
    ----------
    a : TYPE
        Alpha.
    x : TYPE
        Spatial mesh.
    t : TYPE
        Time.

    Returns
    -------
    TYPE
        Analytic solution array.

    """
    return np.exp(-a * np.pi**2 * t) * np.sin(np.pi*x)

def initCond(x):
    """


    Parameters
    ----------
    x : TYPE
        Spatial mesh.

    Returns
    -------
    TYPE
        Initial condition array.

    """
    return np.sin(np.pi*x)


if __name__ == '__main__':
    ### VARIABLES
    a    = 10.0                           # alpha
    xMin = 0
    xMax = 1
    xP   = 51                             # Number of mesh points
    xN   = xP - 1                         # Number of zones
    x    = np.linspace(xMin,xMax,xP)      # Spatial mesh vector
    dx   = x[1] - x[0]
    dxs  = dx**2
    tsc  = 1.0/4.0                        # Time step control parameter
                                          # Note: a*dt/dx**2 = tsc (dimensionless)
    dt   = dxs*tsc/a
    dpa  = -tsc                           # Dimensionless parameter a
    dpb  = 1.0 - 2.0*dpa                  # Dimensionless parameter b
    uBC0 = 0.0                            # Left boundary condition
    uBCN = 0.0                            # Right boundary condition
    u    = np.zeros(xP)                   # Solution vector
    b    = np.zeros(xP)                   # Old solution vector
    # Set initial condition
    b[1:-1] = initCond(x[1:-1])
    # Tridiagonal matrix vectors
    diag = np.zeros(xP)
    lowr = np.zeros(xN)
    uppr = np.zeros(xN)
    # Load matrix vectors
    diag[1:-1] = dpb
    diag[0]    = 1.0
    diag[-1]   = 1.0
    lowr[:-1]  = dpa
    lowr[-1]   = 0.0
    uppr[1:]   = dpa
    uppr[0]    = 0.0
    # Create coefficient matrix
    A = sp.diags(
        diagonals = [diag,lowr,uppr],
        offsets = [0,-1,1],
        shape = (xP,xP),
        format = 'csr')
    # print(A.todense())
    print('dt = {:0.5f}'.format(dt))
    numSteps = 1000
    time = dt*numSteps
    aSol = analSol(a,x,time)
    for i in range(numSteps):
        u[:] = la.spsolve(A,b)
        b, u = u, b
    plt.plot(x,b,'bo',label='Computed')
    plt.plot(x,aSol,label='Analytic',color='red')
    plt.legend()
    plt.title(r'$u_t=\alpha\cdot u_{xx}$')
    plt.text(0,.35,'Time = {:0.5f}s'.format(time))
    plt.show()


