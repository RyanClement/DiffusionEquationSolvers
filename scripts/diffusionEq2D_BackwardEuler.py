#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Program: diffusionEq2D_BackwardEuler
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
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200,precision=1,threshold=10000)


### FUNCTIONS
def initCond(X,Y):
    return np.sin(np.pi*X) * np.sin(np.pi*Y)

def analSol(a,X,Y,t):
    return np.exp(-a * np.pi**2 * 2.0 * t) * np.sin(np.pi*X) * np.sin(np.pi*Y)

def plotSol(X,Y,Z,tit):
    ax   = plt.axes(projection="3d")
    # ax.plot_wireframe(X,Y,Z,color='green')
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='hsv',edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x,y)')
    ax.set_title(tit)
    ax.set_zlim(0,1)
    plt.show()


### MAIN
if __name__ == '__main__':
    ### VARIABLES
    a    = 10.0                           # alpha
    dMin = 0                              # Domain min (square mesh)
    dMax = 1                              # Domain max (square mesh)
    nP   = 51                             # Number of mesh points in each domain coordinate
    nDP  = nP*nP                          # Number of domain points
    nZ   = nP - 1                         # Number of zones in each domain coordinate
    x    = np.linspace(dMin,dMax,nP)      # Spatial mesh vector
    dx   = x[1] - x[0]                    # dx = dy
    dxs  = dx**2
    tsc  = 1.0/4.0                        # Time step control parameter
                                          # Note: a*dt/dx**2 = tsc (dimensionless)
    dt   = dxs*tsc/a
    dpa  = -tsc                           # Dimensionless parameter a
    dpb  = 1.0 + 4.0*tsc                  # Dimensionless parameter b
    b    = np.zeros(nDP)                  # Old solution
    c    = np.zeros(nDP)                  # Solution vector

    # Matrix to vector mapping function
    mat2Vec = lambda i,j: j*nP + i

    # Set initial condition
    X, Y = np.meshgrid(x,x)
    sM = initCond(X,Y)
    # Clean up boundary
    sM[0,:]  = 0.0
    sM[-1,:] = 0.0
    sM[:,0]  = 0.0
    sM[:,-1] = 0.0
    plotSol(X, Y, sM, 'Initial Condition: time = 0.0s')

    # Map sM to b
    # for j in range(nP):
    #     for i in range(nP):
    #         b[mat2Vec(i,j)] = sM[i,j]
    for i in range(nP):
        b[i*nP:(i+1)*nP] = sM[i,:]

    # Construct coefficient matrix
    #  ##### Slow Method #####
    # A    = np.zeros((nDP,nDP))            # Coefficient matrix
    # ## Boundaries
    # for i in range(nP):
    #     p = mat2Vec(i,0);  A[p,p] = 1.0
    #     p = mat2Vec(i,nZ); A[p,p] = 1.0
    #     p = mat2Vec(0,i);  A[p,p] = 1.0
    #     p = mat2Vec(nZ,i); A[p,p] = 1.0
    # ## Internal
    # for j in range(1,nZ):
    #     for i in range(1,nZ):
    #         p = mat2Vec(i,j)
    #         A[p,p]              = dpb
    #         A[p,mat2Vec(i+1,j)] = dpa
    #         A[p,mat2Vec(i,j+1)] = dpa
    #         A[p,mat2Vec(i-1,j)] = dpa
    #         A[p,mat2Vec(i,j-1)] = dpa
    # # Solve
    # numSteps = 100
    # for i in range(numSteps):
    #     c = la.solve(A,b)
    #     b, c = c, b
    ##### Sparce Method #####
    mDiag  = np.zeros(nDP)               # Main diagonal
    l1Diag = np.zeros(nDP-1)             # Lower diagonal #1 (directly below main)
    u1Diag = np.zeros(nDP-1)             # Upper diagonal #1 (directly above main)
    l2Diag = np.zeros(nDP-nP)            # Lower diagonal #2 (starts at row nP)
    u2Diag = np.zeros(nDP-nP)            # Upper diagonal #2 (starts at row 0)
    # Fill diagonals
    mDiag[0:nP] = 1.0                             # j=0  boundary edge
    mDiag[mat2Vec(0,nZ):mat2Vec(nZ,nZ)+1] = 1.0   # j=nZ boundary edge
    for j in range(1,nZ):
        mDiag[mat2Vec(0,j)]  = 1.0                # i=0  boundary edge
        mDiag[mat2Vec(nZ,j)] = 1.0                # i=nZ boundary edge
        mDiag[mat2Vec(1,j):mat2Vec(nZ,j)] = dpb
        u1Diag[mat2Vec(1,j):mat2Vec(nZ,j)] = dpa
        u2Diag[mat2Vec(1,j):mat2Vec(nZ,j)] = dpa
        l1Diag[mat2Vec(1,j)-1:mat2Vec(nZ,j)-1] = dpa
        l2Diag[mat2Vec(1,j)-nP:mat2Vec(nZ,j)-nP] = dpa
    # Create A
    A = sp.diags(
        diagonals=[mDiag,u1Diag,u2Diag,l1Diag,l2Diag],
        offsets=[0,1,nP,-1,-nP],
        shape=(nDP,nDP),
        format='csr')
    numSteps = 500
    for i in range(numSteps):
        c = sla.spsolve(A,b)
        b, c = c, b

    # Plot
    for i in range(nP):
          sM[i,:] = b[i*nP:(i+1)*nP]
    time = dt*numSteps
    plotSol(X,Y,sM,'Computed Solution: time = {:0.5f}s'.format(time))
    aS = analSol(a, X, Y, time)
    plotSol(X,Y,aS,'Analytic Solution: time = {:0.5f}s'.format(time))






















