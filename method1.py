#Method1 : Modified Newton Method

import numpy as np
from tools import *
#  rho, btmax, ro, c1, parameters of armijo condition to add in tools.py and to be imported
def modified_newton(f, gradf, Hessf, x0, kmax, tolgrad=1e-5,rho = 0.55 ,c1 = 0.4, c2 = 0.7, btmax = 10,condition_type = ['armijo','wolfe']):
    
    """
    Modified Newton method for unconstrained optimization.

    Args:
        f : The objective function to minimize.
        gradf : The gradient function of the objective function.
        Hessf : The Hessian function of the objective function.
        x0 : Starting point/initial guess.
        kmax : Maximum number of iterations.
        tolgrad : Tolerance of the gradient of f to use as a stopping criterion.
        rho
        c1
        c2
        condition_type : two options for backtracking either armijo or wolfe

    Returns:
        xk : Final solution.
        f_val : Minimum function value reached at xk.
        k : Number of iterations performed.
        x_seq : Vector containing the points generated at each iteration.
        f_vals : Vector containing the value of f at each generated point at each iteration.
    """
    
    x_seq, f_vals = [], []
    k = 0
    xk = x0
    x_seq.append(xk)
    f_vals.append(f(xk))
    n = np.shape(x0)[0]
    alpha = 1
    
    while k < kmax:
        gradfk = gradf(xk)
        gradfk_norm = np.linalg.norm(gradfk)

        if gradfk_norm < tolgrad:
            print("===Stopping criterion of Tolerance of the gradient satisfied===")
            break
        
        Hk = Hessf(xk)
        beta = np.linalg.norm(Hk)

        Hii = np.diag(Hk)
        if np.all(Hii > 0):
            tau = 0
        else:
            tau = beta / 2

        Bk = Hk + tau * np.eye(n)  

        done = False
        while not done:
            try:
                L = np.linalg.cholesky(Bk)
                done = True
            except:
                #print("Cholesky failed")
                tau = max(2 * tau, beta / 2) 
        while backtrackiter <  btmax:
            if backtrack_condition(condition_type,f,xk,alpha,c1,c2,gradfk,pk):
                break
            alpha *= rho
            backtrackiter += 1

        xk = xk + alpha * pk
        x_seq.append(xk)
        f_vals.append(f(xk))
        k += 1

    if k == kmax:
        print("The optimization stopped because we reached the maximum number of iterations")

    return xk, f(xk), k, x_seq, f_vals
