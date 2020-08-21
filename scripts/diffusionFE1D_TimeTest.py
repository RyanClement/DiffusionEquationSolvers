#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionFE1D_TimeTest
Created: Aug 2020
@author: Ryan Clement (RRCC)
         scisoft@outlook.com

Solve the partial differential equation (PDE)

            u_t = alpha * u_xx

on the interval (0,L) with boundary conditions

            u = 10 for x = 0
            and
            u = 0 for x = 1.0

and initial condition

            u(x,0) = 0.0

The purpose of this program is to test the performance difference
between two methods of coding the same algorithm using a Forward
Euler (explicit) algorithm. Numpy will be used in both cases to
create arrays as we are not interested in timing differnt methods
of list or array creation, e.g. u = n*[0.0] vs u = numpy.zeros(n).
"""


### IMPORTS
import numpy as np
from time import perf_counter as sTime


def solveLoop(n):
    ### VARIABLES
    x0   = 0.0                        # Minumum x-position (left boundary)
    xN   = 1.0                        # Maximum x-position (right boundary)
    xPts = 101                        # Number of spatial mesh points
    xPtsM1 = xPts - 1
    dc   = 0.01                       # Diffusion coefficient
    dx   = (xN - x0)/xPtsM1           # Spatial mesh interval (distance betwen grid points)
    dt   = dx**2/(4*dc)               # Temporal mesh interval (time between solution updates)
    dp   = dc*dt/dx**2                # Dimensionless parameter
    u    = np.zeros(xPts)             # Updated solution array
    uO   = np.zeros(xPts)             # Previous time-step value array
    lBC  = 10.0                       # Left boundary condition
    rBC  = 0.0                        # Right boundary condition

    for j in range(n):
        for i in range(1,xPtsM1):
            u[i] = (1.0 - 2.0*dp)*uO[i] + dp*(uO[i-1] + uO[i+1])
            # Enforce boundary conditions
            u[0]  = lBC
            u[-1] = rBC
        uO[:] = u


def solveVec(n):
    ### VARIABLES
    x0   = 0.0                        # Minumum x-position (left boundary)
    xN   = 1.0                        # Maximum x-position (right boundary)
    xPts = 101                        # Number of spatial mesh points
    xPtsM1 = xPts - 1
    dc   = 0.01                       # Diffusion coefficient
    dx   = (xN - x0)/xPtsM1           # Spatial mesh interval (distance betwen grid points)
    dt   = dx**2/(4*dc)               # Temporal mesh interval (time between solution updates)
    dp   = dc*dt/dx**2                # Dimensionless parameter
    u    = np.zeros(xPts)             # Updated solution array
    uO   = np.zeros(xPts)             # Previous time-step value array
    lBC  = 10.0                       # Left boundary condition
    rBC  = 0.0                        # Right boundary condition

    for j in range(n):
        u[1:xPtsM1] = (1.0 - 2.0*dp)*uO[1:xPtsM1] + dp*(uO[:xPts-2] + uO[2:])
        # Enforce boundary conditions
        u[0]  = lBC
        u[-1] = rBC
        uO, u = u, uO


n = 10000   # Number of time steps

startTimeL = sTime()
solveLoop(n)
stopTimeL = sTime()
totalTimeL = stopTimeL - startTimeL
print("LOOP total elapsed time = {:0.6f} s".format(totalTimeL))

startTimeV = sTime()
solveVec(n)
stopTimeV = sTime()
totalTimeV = stopTimeV - startTimeV
print("VEC total elapsed time = {:0.6f} s".format(totalTimeV))

ratio = totalTimeL / totalTimeV
print("VEC is approximately {:0.8f} faster than LOOP".format(ratio))
