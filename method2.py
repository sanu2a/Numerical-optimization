#Method2 : modified newton method with finite differences 

import numpy as np
from tools import *


#  rho, btmax, c1, parameters of armijo condition to add in tools.py and to be imported
def modified_newton_FD(f, x0, kmax, fd="c", tolgrad=1e-5, rho = 0.4 ,c1 = 1e-4, c2 = 0.7, btmax = 10,condition_type = 'armijo'):
    """
    Modified Newton method with finite differences.

    Args:
        f : The objective function to minimize.
        x0 : Starting point/initial guess.
        kmax : Maximum number of iterations.
        fd : Finite differences method ('c' for central, 'f' for forward, 'None' for default).
        tolgrad : Tolerance of the gradient of f to use as a stopping criterion.

    Returns:
        xk : Final solution.
        f_val : Minimum function value reached at xk.
        k : Number of iterations performed.
        x_seq : Vector containing the points generated at each iteration.
        f_vals : Vector containing the value of f at each generated point at each iteration.
    """
    
    hgrad = np.sqrt(np.finfo(float).eps)
    hhess = np.sqrt(hgrad)
    gradf_fd = grad_CFD
    hessf_fd = hessf_FD

    if fd == "FW":
        gradf_fd = grad_FWFD

    x_seq, f_vals = [], []
    k = 0
    xk = x0
    x_seq.append(xk)
    f_vals.append(f(xk))
    n = np.shape(x0)[0]
    alpha = 1

    while k < kmax:
        gradfk = gradf_fd(f, xk, hgrad)
        gradfk_norm = np.linalg.norm(gradfk)

        if gradfk_norm < tolgrad:
            print("===Stopping criterion of Tolerance of the gradient satisfied===")
            break

        Hk = hessf_fd(f, xk, hhess)
        beta = np.linalg.norm(Hk)
        Hii = np.diag(Hk)
        if np.all(Hii > 0):
            tau = 0
        else:
            tau = beta / 2
            
        done = False
        while not done:
            Bk = Hk + tau * np.eye(n)
            try:
                L = np.linalg.cholesky(Bk)
                done = True
            except:
                #print("Cholesky failed")
                tau = max(2 * tau, beta / 2)
                
        y = np.linalg.solve(L, -gradfk)
        pk = np.linalg.solve(L.T, y)
        
        backtrackiter = 0
        while backtrackiter <  btmax:
            if backtrack_condition(condition_type,f,xk,alpha,c1,c2,gradfk,pk):
                #print("Backtracking satisfied")
                break
            alpha *= rho
            backtrackiter += 1
        if backtrackiter == btmax : 
            print("Max backtrack reached")
        xk = xk + alpha * pk
        x_seq.append(xk)
        f_vals.append(f(xk))
        k += 1

    if k == kmax:
        print("The optimization stopped because we reached the maximum number of iterations")

    return xk, f(xk), k, x_seq, f_vals
