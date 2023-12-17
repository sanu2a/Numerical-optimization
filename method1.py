#Method1

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import *
import sys
def LineSearch(f,x,alpha,p):
    # f is function 
    # current x value
    # p is direction
    condition = np.dot(f(x+alpha*p),p) == 0
    return condition

def modified_newton(f,g,H,x0,kmax,rho,tolgrad):
    # f is the function
    # g is the gradient
    # h is hessian
    x0_vals = []
    x1_vals = []
    f_vals = [] #evaluation
    x0_vals.append(x0[0])
    x1_vals.append(x0[1])
    f_vals.append(f(x0))
    
    n = np.shape(x0)[0]   #number of vars as the number of initilazers
    sigma = 0.4
    tau = 0.0 # how muh i want to regularize the hessian
    k = 0
    epsilon = 1e-5
    while k<kmax:
        gk = g(x0) 
        gk_norm = np.linalg.norm(gk) # Calculate the norm of the gradient vector
        if gk_norm < epsilon:
            break
        muk = np.power(np.linalg.norm(gk),1+tau) # mu k is the amount i want to push all the eigenvalues of the hessian
        Hk = H(x0)
        Ak = Hk + muk*np.eye(n)     # regularize by adding identity matrix
        dk = -1.0*np.linalg.solve(Ak,gk) # finding the direction
        m = 0
        mk = 0
        while m<20:
            if LineSearch(g,x0,rho**m,dk):
                #rho**m shrinks it until the line search condition in satisfied
                mk = m
                break
            m +=1
        x0 += rho**mk*dk
        x0_vals.append(x0[0])
        x1_vals.append(x0[1])
        f_vals.append(f(x0))
        k+=1
    print("========= modified newton method with line search")
    print("xk = ",x0)
    print("f(xk) = ",round(f(x0),3))
    print("number of iterations = ",k)
    return x0_vals,x1_vals,f_vals