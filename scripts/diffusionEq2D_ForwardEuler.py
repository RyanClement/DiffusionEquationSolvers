#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionEq2D_ForwardEulerV
Created: Aug 2020
@author: Ryan Clement (RRCC)
         scisoft@outlook.com

Purpose: Solve the partial differential equation (PDE)

            u_t = alpha * (u_xx + u_yy)

in (0,Lx)x(0,Ly) with vanishing boundary conditions or

            u = 0 for x = 0, y in [0,Ly]
            u = 0 for x = 1.0, y in [0,Ly]
            u = 0 for y = 0, x in [0,Lx]
            u = 0 for y = 1.0, x in [0,Lx]

and initial condition

            u(x,y,0) = I(x,y) = A*sin(Pi*x/Lx)*sin(Pi*y/Ly)

The analytic solution for this problem is given by

            u(x,y,t) = Ae**(-alpha*Pi**2*(Lx**-2 + Ly**-2)*t)*sin(Pi*x/Lx)*sin(Pi*y/Ly)

We will take A=Lx=Ly=1 for this simulation
"""


### IMPORTS
import numpy as np
import matplotlib.pyplot as plt


### FUNCTIONS
def initialCondition(xN,yN,X,Y):
    return np.sin(np.pi*X/xN) * np.sin(np.pi*Y/yN)

def analSol(a,xN,yN,X,Y,t):
    return np.exp(-a * np.pi**2 * (1.0/xN**2 + 1.0/yN**2) * t) * np.sin(np.pi*X/xN) * np.sin(np.pi*Y/yN)

def plotInitialCondition(a,xN,yN,X,Y):
    # fig  = plt.figure()
    ax   = plt.axes(projection="3d")
    Z    = initialCondition(xN,yN,X,Y)
    # ax.plot_wireframe(X,Y,Z,color='green')
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='hsv',edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.set_title('Initial Condition')
    ax.set_zlim(0,1)
    plt.show()

def plotSol(X,Y,u,tit):
    ax   = plt.axes(projection="3d")
    # ax.plot_wireframe(X,Y,u,color='green')
    ax.plot_surface(X,Y,u,rstride=1,cstride=1,cmap='hsv',edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.set_title(tit)
    ax.set_zlim(0,1)
    plt.show()

def plotDiff(X,Y,u,tit):
    ax   = plt.axes(projection="3d")
    ax.plot_wireframe(X,Y,u,color='green')
    # ax.plot_surface(X,Y,u,rstride=1,cstride=1,cmap='winter',edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\frac{\Delta u}{u}$')
    ax.set_title(tit)
    plt.show()

def setIC_Loop(a,x,y,u,xPts,yPts,xN,yN):
    for j in range(yPts):
        for i in range(xPts):
            u[i,j] = initialCondition(a, xN, yN, x[i], y[j])

def setIC(xN,yN,X,Y):
    u = initialCondition(xN, yN, X, Y)
    # Clean up boundary
    u[0,:]  = 0.0
    u[-1,:] = 0.0
    u[:,0]  = 0.0
    u[:,-1] = 0.0
    return u

def advanceN(a,xN,yN,u,n,xPts,yPts):
    ### VARIABLES
    x0   = 0.0                        # X: Minimum value (Left boundary)
    dx   = (xN - x0)/(xPts-1)         # X: Distance between mesh points
    dxs  = dx**2
    y0   = 0.0                        # Y: Minimum value (Bottom boundary)
    dy   = (yN - y0)/(yPts-1)         # Y: Distance between mesh points
    dys  = dy**2
    tsc  = 10                         # Time step control parameter
    dtx  = dxs/(tsc*a)                # X: Temporal mesh interval
    dty  = dys/(tsc*a)                # Y: Temporal mesh interval
    dt   = min(dtx, dty)
    at   = a*dt
    dpx  = at/dxs                     # Dimensionless parameter
    dpy  = at/dys                     # Dimensionless parameter
    uO   = np.copy(u)                 # Previous time-step value array (shallow copy)
    for k in range(n):
        for j in range(1,yPts-1):
            for i in range(1,xPts-1):
                u[i,j] = uO[i,j] + \
                         dpx*(uO[i+1,j] - 2.0*uO[i,j] + uO[i-1,j]) + \
                         dpy*(uO[i,j+1] - 2.0*uO[i,j] + uO[i,j-1])
        # Boundary Condition
        u[0,:]  = 0.0
        u[-1,:] = 0.0
        u[:,0]  = 0.0
        u[:,-1] = 0.0
        uO, u = u, uO
    return uO,dt



### MAIN
a    = 1.0
xN   = 1.0
yN   = 1.0
xPts = 51
yPts = 51
x    = np.linspace(0,xN,xPts)
y    = np.linspace(0,yN,yPts)
X, Y = np.meshgrid(x,y)
u = setIC(xN,yN,X,Y)
plotSol(X,Y,u,'Initial Condition')
numSteps = 1000
uS,dt = advanceN(a,xN,yN,u,numSteps,xPts,yPts)
print('Time Step = {:0.5f} s'.format(dt))
time = numSteps*dt
print('Simulation End Time = {:0.5f} s'.format(time))
plotSol(X,Y,uS,'Simulation Time {:.5f} s'.format(time))
uA = analSol(a,xN,yN,X,Y,time)
plotSol(X,Y,uA,'Analytic Time {:.5f} s'.format(time))
uFracDiff = np.copy(uA)
for j in range(yPts):
    for i in range(xPts):
        ua = uA[i,j]
        us = uS[i,j]
        if 0.0 == ua or 0.0 == us:
            uFracDiff[i,j] = 0.0
        else:
            uFracDiff[i,j] = ua/us - 1.0
plotDiff(X,Y,uFracDiff,'Fractional Difference')

